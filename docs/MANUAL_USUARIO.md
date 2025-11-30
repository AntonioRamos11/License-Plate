# 📖 Manual de Usuario
## Sistema de Detección de Matrículas con Identificación de Propietarios

---

## 📑 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación](#instalación)
4. [Guía de Uso](#guía-de-uso)
5. [Gestión de Base de Datos](#gestión-de-base-de-datos)
6. [Solución de Problemas](#solución-de-problemas)
7. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## 🎯 Introducción

Este sistema permite detectar automáticamente las matrículas de vehículos en imágenes y videos, identificando al propietario del vehículo mediante una base de datos integrada.

### Características Principales
- ✅ Detección automática de matrículas usando IA (YOLOv5)
- ✅ Identificación de propietarios de vehículos
- ✅ Base de datos integrada con información de vehículos
- ✅ Procesamiento de imágenes y videos
- ✅ Historial de detecciones
- ✅ Interfaz visual de resultados

---

## 💻 Requisitos del Sistema

### Hardware Mínimo
- **CPU**: Intel Core i5 o equivalente
- **RAM**: 8 GB mínimo (16 GB recomendado)
- **Almacenamiento**: 5 GB de espacio libre
- **GPU** (opcional): NVIDIA con soporte CUDA para mejor rendimiento

### Software
- **Sistema Operativo**: Windows 10/11, Linux (Ubuntu 18.04+), o macOS 10.15+
- **Python**: Versión 3.7 o superior
- **CUDA** (opcional): Para aceleración GPU

---

## 🔧 Instalación

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/zeusees/License-Plate-Detector.git
cd License-Plate-Detector
```

### Paso 2: Crear Entorno Virtual (Recomendado)

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar el Modelo Preentrenado

El modelo preentrenado debe estar en la carpeta `weights/best.pt`. Si no está presente:

```bash
cd weights
bash download_weights.sh
cd ..
```

### Paso 5: Verificar Instalación

```bash
python detect_owner.py --help
```

Si ve el menú de ayuda, ¡la instalación fue exitosa! ✅

---

## 🚀 Guía de Uso

### Uso Básico

#### 1. Detectar Matrículas en una Imagen

```bash
python detect_owner.py --source ruta/a/tu/imagen.jpg
```

**Ejemplo:**
```bash
python detect_owner.py --source imgs/test_car.jpg
```

#### 2. Detectar Matrículas en un Video

```bash
python detect_owner.py --source ruta/a/tu/video.mp4
```

**Ejemplo:**
```bash
python detect_owner.py --source videos/traffic.mp4 --output results/output.mp4
```

### Opciones Avanzadas

#### Guardar Resultados

```bash
python detect_owner.py --source imagen.jpg --output resultados/imagen_detectada.jpg
```

#### Ajustar Umbral de Confianza

```bash
python detect_owner.py --source imagen.jpg --conf-thres 0.5
```
*Valores más altos = menos detecciones pero más precisas*

#### Usar GPU Específica

```bash
python detect_owner.py --source imagen.jpg --device 0
```
*0 = primera GPU, 1 = segunda GPU, etc.*

#### Usar CPU Solamente

```bash
python detect_owner.py --source imagen.jpg --device cpu
```

#### No Mostrar Ventana de Resultados

```bash
python detect_owner.py --source imagen.jpg --no-show --output resultado.jpg
```

### Ejemplos Completos

**Ejemplo 1: Procesar imagen con alta precisión**
```bash
python detect_owner.py \
    --source imgs/parking.jpg \
    --conf-thres 0.6 \
    --output results/parking_detected.jpg
```

**Ejemplo 2: Procesar video y guardarlo**
```bash
python detect_owner.py \
    --source videos/highway.mp4 \
    --output results/highway_detected.mp4 \
    --conf-thres 0.4
```

**Ejemplo 3: Procesamiento sin visualización (para servidores)**
```bash
python detect_owner.py \
    --source imagen.jpg \
    --no-show \
    --output resultado.jpg \
    --device cpu
```

---

## 🗄️ Gestión de Base de Datos

### Inicializar la Base de Datos

La base de datos se crea automáticamente la primera vez que ejecuta el sistema. Para gestionarla manualmente:

```bash
python database/vehicle_database.py
```

### Agregar Propietarios y Vehículos

Puede agregar datos usando el script interactivo:

```python
from database.vehicle_database import VehicleDatabase

# Crear conexión
db = VehicleDatabase()

# Agregar propietario
propietario_id = db.agregar_propietario(
    nombre="Juan",
    apellido="Pérez",
    dni="12345678A",
    telefono="+34 600123456",
    email="juan@email.com",
    direccion="Calle Mayor 1, Madrid"
)

# Agregar vehículo
vehiculo_id = db.agregar_vehiculo(
    matricula="1234ABC",
    marca="Toyota",
    modelo="Corolla",
    anio=2020,
    color="Blanco",
    propietario_id=propietario_id
)

# Cerrar conexión
db.close()
```

### Buscar Propietario por Matrícula

```python
from database.vehicle_database import VehicleDatabase

db = VehicleDatabase()
resultado = db.buscar_propietario_por_matricula("1234ABC")

if resultado:
    print(f"Propietario: {resultado['propietario']['nombre_completo']}")
    print(f"Vehículo: {resultado['vehiculo']['marca']} {resultado['vehiculo']['modelo']}")

db.close()
```

### Ver Historial de Detecciones

```python
from database.vehicle_database import VehicleDatabase

db = VehicleDatabase()
historial = db.obtener_historial_vehiculo("1234ABC", limit=10)

for deteccion in historial:
    print(f"Fecha: {deteccion['fecha']}")
    print(f"Ubicación: {deteccion['ubicacion']}")
    print(f"Confianza: {deteccion['confianza']:.2%}")

db.close()
```

---

## 🔧 Solución de Problemas

### Problema: "No se pudo cargar el modelo"

**Solución:**
1. Verifique que el archivo `weights/best.pt` existe
2. Si no existe, descárguelo ejecutando:
   ```bash
   cd weights
   bash download_weights.sh
   ```

### Problema: "CUDA out of memory"

**Solución:**
1. Use un tamaño de imagen más pequeño:
   ```bash
   python detect_owner.py --source imagen.jpg --img-size 416
   ```
2. O use CPU en lugar de GPU:
   ```bash
   python detect_owner.py --source imagen.jpg --device cpu
   ```

### Problema: "No se detectan matrículas"

**Solución:**
1. Reduzca el umbral de confianza:
   ```bash
   python detect_owner.py --source imagen.jpg --conf-thres 0.1
   ```
2. Verifique que la imagen tiene buena calidad y las matrículas son visibles
3. Asegúrese de que la imagen no esté muy oscura o borrosa

### Problema: "Propietario no encontrado"

**Solución:**
- El sistema detectó la matrícula pero no está registrada en la base de datos
- Agregue el vehículo y propietario a la base de datos siguiendo la [Guía de Gestión de BD](#gestión-de-base-de-datos)

### Problema: Error de importación de módulos

**Solución:**
```bash
pip install -r requirements.txt --upgrade
```

### Problema: El video no se procesa

**Solución:**
1. Verifique que OpenCV está instalado correctamente:
   ```bash
   pip install opencv-python --upgrade
   ```
2. Intente con un formato de video diferente (MP4, AVI)
3. Asegúrese de que el archivo de video no está corrupto

---

## ❓ Preguntas Frecuentes

### ¿Qué formatos de imagen soporta el sistema?
El sistema soporta los siguientes formatos:
- JPG/JPEG
- PNG
- BMP
- TIFF

### ¿Qué formatos de video soporta?
- MP4
- AVI
- MOV
- MKV

### ¿Puedo procesar múltiples imágenes a la vez?
Actualmente el sistema procesa una imagen o video a la vez. Para procesar múltiples archivos, puede crear un script bash:

```bash
#!/bin/bash
for img in imgs/*.jpg; do
    python detect_owner.py --source "$img" --output "results/$(basename $img)"
done
```

### ¿Cómo mejoro la precisión de detección?
1. Use imágenes de alta calidad
2. Asegúrese de buena iluminación
3. Las matrículas deben estar frontales (no muy inclinadas)
4. Ajuste el umbral de confianza según sus necesidades
5. Use GPU para mejor rendimiento

### ¿El sistema funciona en tiempo real?
Sí, puede procesar video en tiempo real si tiene una GPU NVIDIA con CUDA. El rendimiento depende de:
- Potencia de su GPU/CPU
- Resolución del video
- Tamaño de inferencia configurado

### ¿Cómo exporto la base de datos?
La base de datos SQLite está en `database/vehicles.db`. Puede:
1. Copiar este archivo para hacer un backup
2. Usar herramientas como DB Browser for SQLite para ver/exportar datos
3. Usar scripts Python para exportar a CSV/JSON

### ¿Puedo entrenar el modelo con mis propias imágenes?
Sí, puede entrenar el modelo usando el script `train.py`. Consulte la [Documentación Técnica](docs/DOCUMENTACION_TECNICA.md) para más detalles.

### ¿El sistema requiere internet?
No, una vez instalado, el sistema funciona completamente offline.

### ¿Cómo actualizo el sistema?
```bash
git pull origin master
pip install -r requirements.txt --upgrade
```

---

## 📞 Soporte

Si tiene problemas no cubiertos en esta guía:

1. **Revise los logs**: El sistema imprime mensajes detallados en consola
2. **Consulte la documentación técnica**: `docs/DOCUMENTACION_TECNICA.md`
3. **Reporte un issue**: En el repositorio de GitHub
4. **Contacto**: Consulte el README principal para información de contacto

---

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el repositorio principal.

---

**Última actualización:** Noviembre 2025
**Versión del manual:** 1.0
