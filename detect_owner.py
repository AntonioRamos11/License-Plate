"""
Sistema Integral de Detección de Matrículas con Identificación de Propietarios
Este script integra el modelo YOLOv5 de detección de matrículas con la base de datos
de propietarios de vehículos.
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import os
import time
import easyocr
import re
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.experimental import attempt_load
from utils.general import non_max_suppression_plate, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box
from database.vehicle_database import VehicleDatabase


class LicensePlateDetector:
    """Clase para detectar matrículas e identificar propietarios."""
    
    def __init__(self, weights='weights/best.pt', img_size=640, conf_thres=0.25, 
                 iou_thres=0.45, device='', db_path='database/vehicles.db'):
        """
        Inicializa el detector de matrículas.
        
        Args:
            weights (str): Ruta al archivo de pesos del modelo
            img_size (int): Tamaño de imagen para inferencia
            conf_thres (float): Umbral de confianza para detección
            iou_thres (float): Umbral de IoU para NMS
            device (str): Dispositivo CUDA (ej: '0' o 'cpu')
            db_path (str): Ruta a la base de datos de vehículos
        """
        print("🚀 Inicializando Sistema de Detección de Matrículas...")
        
        # Configurar logging
        set_logging()
        
        # Seleccionar dispositivo
        self.device = select_device(device)
        print(f"📱 Dispositivo seleccionado: {self.device}")
        
        # Cargar modelo
        print(f"🔄 Cargando modelo desde {weights}...")
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Configurar half precision si es posible
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        
        # Nombres de las clases
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Colores para visualización
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        
        # Inicializar EasyOCR
        print("🔤 Inicializando motor OCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Inicializar base de datos
        print(f"💾 Conectando a base de datos: {db_path}")
        self.db = VehicleDatabase(db_path)
        
        # Ejecutar warmup
        print("🔥 Ejecutando warmup del modelo...")
        img = torch.zeros((1, 3, img_size, img_size), device=self.device)
        _ = self.model(img.half() if self.half else img)
        
        print("✅ Sistema inicializado correctamente\n")
    
    def preprocess_image(self, img0):
        """
        Preprocesa la imagen para inferencia.
        
        Args:
            img0 (numpy.ndarray): Imagen original
            
        Returns:
            tuple: (imagen procesada, imagen original, dimensiones originales)
        """
        # Redimensionar y pad
        img = cv2.resize(img0, (self.img_size, self.img_size))
        
        # Convertir BGR a RGB y reorganizar dimensiones
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR a RGB, HWC a CHW
        img = np.ascontiguousarray(img)
        
        # Convertir a tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # Normalizar 0-255 a 0.0-1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, img0, img0.shape
    
    def detect_plate(self, img0):
        """
        Detecta matrículas en una imagen.
        
        Args:
            img0 (numpy.ndarray): Imagen original
            
        Returns:
            list: Lista de detecciones con coordenadas, confianza y clase
        """
        # Preprocesar imagen
        img, img0, img0_shape = self.preprocess_image(img0)
        
        # Inferencia
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        
        # Aplicar NMS
        pred = non_max_suppression_plate(pred, self.conf_thres, self.iou_thres)
        t2 = time_synchronized()
        
        # Procesar detecciones
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                # Reescalar coordenadas a tamaño original
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
                
                # Convertir a lista de diccionarios
                for *xyxy, conf, cls in det:
                    detection = {
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.names[int(cls)]
                    }
                    detections.append(detection)
        
        inference_time = (t2 - t1) * 1000  # en milisegundos
        
        return detections, inference_time
    
    def extract_plate_text(self, img, bbox):
        """
        Extrae el texto de una matrícula detectada usando OCR.
        
        Args:
            img (numpy.ndarray): Imagen original
            bbox (list): Coordenadas [x1, y1, x2, y2]
            
        Returns:
            str: Texto extraído de la matrícula
        """
        x1, y1, x2, y2 = bbox
        
        # Agregar margen para mejorar la detección
        margin = 5
        h, w = img.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Recortar región de la matrícula
        plate_img = img[y1:y2, x1:x2]
        
        # Preprocesar imagen para mejorar OCR
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate_gray = cv2.GaussianBlur(plate_gray, (3, 3), 0)
        plate_gray = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Realizar OCR
        try:
            results = self.reader.readtext(plate_gray, detail=0, paragraph=False)
            
            if results:
                # Concatenar todos los textos detectados
                plate_text = ''.join(results)
                # Limpiar texto: solo letras y números
                plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
                return plate_text if plate_text else "DESCONOCIDO"
            else:
                return "DESCONOCIDO"
        except Exception as e:
            print(f"    ⚠️  Error en OCR: {e}")
            return "ERROR_OCR"
    
    def identify_owner(self, plate_text):
        """
        Identifica al propietario de un vehículo por su matrícula.
        
        Args:
            plate_text (str): Texto de la matrícula
            
        Returns:
            dict: Información del propietario y vehículo, o None
        """
        return self.db.buscar_propietario_por_matricula(plate_text)
    
    def draw_detection(self, img, detection, owner_info=None):
        """
        Dibuja la detección en la imagen.
        
        Args:
            img (numpy.ndarray): Imagen donde dibujar
            detection (dict): Información de la detección
            owner_info (dict, optional): Información del propietario
            
        Returns:
            numpy.ndarray: Imagen con detección dibujada
        """
        bbox = detection['bbox']
        conf = detection['confidence']
        cls_name = detection['class_name']
        
        # Determinar color
        color = self.colors[detection['class']]
        
        # Dibujar bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto
        if owner_info:
            label = f"{cls_name} {conf:.2f}"
            owner_label = f"Propietario: {owner_info['propietario']['nombre_completo']}"
            vehicle_label = f"Vehiculo: {owner_info['vehiculo']['marca']} {owner_info['vehiculo']['modelo']}"
            
            # Dibujar etiquetas
            self._draw_label(img, label, (x1, y1 - 10), color)
            self._draw_label(img, owner_label, (x1, y2 + 20), (0, 255, 0))
            self._draw_label(img, vehicle_label, (x1, y2 + 45), (0, 255, 0))
        else:
            label = f"{cls_name} {conf:.2f} - Desconocido"
            self._draw_label(img, label, (x1, y1 - 10), color)
        
        return img
    
    def _draw_label(self, img, text, position, color):
        """Dibuja una etiqueta con fondo en la imagen."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Obtener tamaño del texto
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Dibujar fondo
        x, y = position
        cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), color, -1)
        
        # Dibujar texto
        cv2.putText(img, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    def process_image(self, image_path, output_path=None, show=False):
        """
        Procesa una imagen completa: detecta, identifica y visualiza.
        
        Args:
            image_path (str): Ruta a la imagen de entrada
            output_path (str, optional): Ruta para guardar resultado
            show (bool): Si mostrar la imagen en ventana
            
        Returns:
            list: Lista de detecciones con información de propietarios
        """
        print(f"\n🖼️  Procesando imagen: {image_path}")
        
        # Cargar imagen
        img0 = cv2.imread(str(image_path))
        if img0 is None:
            print(f"❌ Error: No se pudo cargar la imagen {image_path}")
            return []
        
        # Detectar matrículas
        detections, inference_time = self.detect_plate(img0)
        print(f"⏱️  Tiempo de inferencia: {inference_time:.1f}ms")
        print(f"🔍 Detecciones encontradas: {len(detections)}")
        
        # Procesar cada detección
        results = []
        for i, det in enumerate(detections):
            print(f"\n  Detección {i+1}:")
            print(f"    Confianza: {det['confidence']*100:.2f}%")
            
            # Extraer texto de matrícula (simulado)
            plate_text = self.extract_plate_text(img0, det['bbox'])
            print(f"    Matrícula: {plate_text}")
            
            # Buscar propietario
            owner_info = self.identify_owner(plate_text)
            
            if owner_info:
                print(f"    ✅ Propietario encontrado: {owner_info['propietario']['nombre_completo']}")
                print(f"    Vehículo: {owner_info['vehiculo']['marca']} {owner_info['vehiculo']['modelo']}")
                
                # Registrar detección
                self.db.registrar_deteccion(
                    vehiculo_id=owner_info['vehiculo']['id'],
                    ubicacion=str(image_path),
                    confianza=det['confidence'],
                    imagen_path=output_path
                )
            else:
                print(f"    ⚠️  Propietario no encontrado en la base de datos")
            
            # Dibujar detección
            img0 = self.draw_detection(img0, det, owner_info)
            
            results.append({
                'detection': det,
                'plate_text': plate_text,
                'owner': owner_info
            })
        
        # Guardar resultado
        if output_path:
            cv2.imwrite(output_path, img0)
            print(f"\n💾 Resultado guardado en: {output_path}")
        
        # Mostrar resultado
        if show:
            try:
                # Intentar mostrar con matplotlib (funciona mejor sin GUI)
                plt.figure(figsize=(12, 8))
                # Convertir BGR a RGB para matplotlib
                img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.title('Detección de Matrículas', fontsize=16)
                
                # Guardar la visualización
                viz_path = str(Path(output_path).parent / f"{Path(output_path).stem}_viz.png") if output_path else 'detection_result.png'
                plt.savefig(viz_path, bbox_inches='tight', dpi=150)
                print(f"\n📊 Visualización guardada en: {viz_path}")
                plt.close()
            except Exception as e:
                print(f"\n⚠️  No se puede mostrar/guardar la visualización: {e}")
                if output_path:
                    print(f"   Revisa la imagen guardada en: {output_path}")
        
        return results
    
    def process_video(self, video_path, output_path=None, show=True):
        """
        Procesa un video completo detectando matrículas.
        
        Args:
            video_path (str): Ruta al video de entrada
            output_path (str, optional): Ruta para guardar resultado
            show (bool): Si mostrar el video en tiempo real
        """
        print(f"\n🎥 Procesando video: {video_path}")
        
        # Abrir video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir el video {video_path}")
            return
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Propiedades del video:")
        print(f"    Resolución: {width}x{height}")
        print(f"    FPS: {fps}")
        print(f"    Total de frames: {total_frames}")
        
        # Configurar writer si se va a guardar
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Procesar frames
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detectar matrículas
                detections, inference_time = self.detect_plate(frame)
                
                # Procesar detecciones
                for det in detections:
                    detection_count += 1
                    plate_text = self.extract_plate_text(frame, det['bbox'])
                    owner_info = self.identify_owner(plate_text)
                    frame = self.draw_detection(frame, det, owner_info)
                
                # Agregar información del frame
                info_text = f"Frame: {frame_count}/{total_frames} | Detecciones: {len(detections)} | {inference_time:.1f}ms"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Guardar frame
                if writer:
                    writer.write(frame)
                
                # Mostrar frame
                if show:
                    cv2.imshow('Detección de Matrículas - Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Detenido por el usuario")
                        break
                
                # Mostrar progreso
                if frame_count % 30 == 0:
                    print(f"Procesados {frame_count}/{total_frames} frames...", end='\r')
        
        finally:
            # Liberar recursos
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Video procesado completamente")
            print(f"    Frames procesados: {frame_count}")
            print(f"    Total de detecciones: {detection_count}")
            if output_path:
                print(f"    Guardado en: {output_path}")
    
    def __del__(self):
        """Destructor para cerrar la conexión a la base de datos."""
        if hasattr(self, 'db'):
            self.db.close()


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description='Sistema de Detección de Matrículas con Identificación de Propietarios')
    parser.add_argument('--source', type=str, required=True, help='Ruta a imagen o video')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Ruta al modelo')
    parser.add_argument('--img-size', type=int, default=640, help='Tamaño de imagen para inferencia')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Umbral de confianza')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='Umbral IoU para NMS')
    parser.add_argument('--device', default='', help='Dispositivo cuda (ej: 0 o cpu)')
    parser.add_argument('--output', type=str, help='Ruta para guardar resultado')
    parser.add_argument('--no-show', action='store_true', help='No mostrar resultado')
    parser.add_argument('--db-path', type=str, default='database/vehicles.db', 
                       help='Ruta a la base de datos')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = LicensePlateDetector(
        weights=args.weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        db_path=args.db_path
    )
    
    # Determinar tipo de fuente
    source = Path(args.source)
    
    if not source.exists():
        print(f"❌ Error: La fuente {source} no existe")
        return
    
    # Procesar según tipo
    if source.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Es una imagen
        # Generar ruta de salida si no se especifica
        if not args.output:
            args.output = str(source.parent / f"{source.stem}_result{source.suffix}")
        
        detector.process_image(
            image_path=str(source),
            output_path=args.output,
            show=not args.no_show
        )
    elif source.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Es un video
        detector.process_video(
            video_path=str(source),
            output_path=args.output,
            show=not args.no_show
        )
    else:
        print(f"❌ Error: Formato no soportado {source.suffix}")
        print("Formatos soportados: imágenes (.jpg, .png) y videos (.mp4, .avi)")


if __name__ == '__main__':
    main()
