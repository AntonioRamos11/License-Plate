# 📚 Documentación Técnica
## Sistema de Detección de Matrículas con Identificación de Propietarios

---

## 📑 Tabla de Contenidos
1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Componentes Principales](#componentes-principales)
3. [Base de Datos](#base-de-datos)
4. [Modelo de Visión Artificial](#modelo-de-visión-artificial)
5. [APIs y Módulos](#apis-y-módulos)
6. [Manual de Instalación](#manual-de-instalación)
7. [Configuración Avanzada](#configuración-avanzada)
8. [Desarrollo y Contribución](#desarrollo-y-contribución)

---

## 🏗️ Arquitectura del Sistema

### Diagrama General

```
┌─────────────────────────────────────────────────────────────┐
│                     USUARIO FINAL                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              INTERFAZ DE LÍNEA DE COMANDOS                   │
│                  (detect_owner.py)                           │
└──────────────┬─────────────────────┬────────────────────────┘
               │                     │
               ▼                     ▼
┌──────────────────────┐  ┌─────────────────────────────────┐
│  MÓDULO DE DETECCIÓN │  │   MÓDULO DE BASE DE DATOS       │
│    (YOLOv5 Model)    │  │  (vehicle_database.py)          │
│                      │  │                                  │
│  - Carga de modelo   │  │  - Gestión de propietarios      │
│  - Preprocesamiento  │  │  - Gestión de vehículos         │
│  - Inferencia        │  │  - Historial de detecciones     │
│  - Post-procesamiento│  │  - Consultas SQL                │
└──────────────────────┘  └─────────────┬───────────────────┘
               │                         │
               └─────────┬───────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │    SISTEMA DE VINCULACIÓN   │
           │  - Extracción de texto (OCR)│
           │  - Búsqueda en BD           │
           │  - Registro de detecciones  │
           └─────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │   RESULTADOS Y VISUALIZACIÓN│
           │  - Imágenes anotadas        │
           │  - Videos procesados        │
           │  - Logs de detecciones      │
           └─────────────────────────────┘
```

### Flujo de Datos

1. **Entrada**: Usuario proporciona imagen/video
2. **Detección**: YOLOv5 detecta matrículas
3. **Extracción**: Se extrae el texto de la matrícula (OCR simulado)
4. **Búsqueda**: Se busca el propietario en la base de datos
5. **Registro**: Se guarda la detección en el historial
6. **Salida**: Se visualiza/guarda el resultado anotado

---

## 🔧 Componentes Principales

### 1. detect_owner.py
**Propósito**: Script principal que integra todos los componentes

**Clases Principales**:

#### `LicensePlateDetector`
```python
class LicensePlateDetector:
    def __init__(self, weights, img_size, conf_thres, iou_thres, device, db_path)
    def preprocess_image(self, img0)
    def detect_plate(self, img0)
    def extract_plate_text(self, img, bbox)
    def identify_owner(self, plate_text)
    def draw_detection(self, img, detection, owner_info)
    def process_image(self, image_path, output_path, show)
    def process_video(self, video_path, output_path, show)
```

**Características**:
- Carga y gestión del modelo YOLOv5
- Preprocesamiento de imágenes
- Detección de matrículas
- Integración con base de datos
- Visualización de resultados

### 2. database/vehicle_database.py
**Propósito**: Gestión completa de la base de datos

**Clases Principales**:

#### `VehicleDatabase`
```python
class VehicleDatabase:
    def __init__(self, db_path)
    def agregar_propietario(self, nombre, apellido, dni, ...)
    def agregar_vehiculo(self, matricula, marca, modelo, ...)
    def buscar_propietario_por_matricula(self, matricula)
    def registrar_deteccion(self, vehiculo_id, ubicacion, ...)
    def obtener_historial_vehiculo(self, matricula, limit)
    def listar_todos_los_vehiculos(self)
    def actualizar_propietario(self, propietario_id, **kwargs)
    def eliminar_vehiculo(self, matricula)
```

**Características**:
- Gestión de conexiones SQLite
- CRUD completo de propietarios y vehículos
- Historial de detecciones
- Manejo robusto de errores
- Context manager para gestión de recursos

### 3. models/
**Propósito**: Definición de la arquitectura del modelo YOLOv5

**Archivos Clave**:
- `common.py`: Bloques de construcción de la red neuronal
- `yolo.py`: Definición de la arquitectura YOLOv5
- `experimental.py`: Funciones para cargar modelos

### 4. utils/
**Propósito**: Utilidades y funciones auxiliares

**Archivos Clave**:
- `datasets.py`: Carga y preprocesamiento de datos
- `general.py`: Funciones generales (NMS, conversiones)
- `plots.py`: Visualización de resultados
- `torch_utils.py`: Utilidades de PyTorch
- `metrics.py`: Métricas de evaluación

---

## 🗄️ Base de Datos

### Esquema de la Base de Datos

#### Tabla: `propietarios`
```sql
CREATE TABLE propietarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    apellido TEXT NOT NULL,
    dni TEXT UNIQUE NOT NULL,
    telefono TEXT,
    email TEXT,
    direccion TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Campos**:
- `id`: Identificador único autoincremental
- `nombre`: Nombre del propietario
- `apellido`: Apellido del propietario
- `dni`: DNI único (clave única)
- `telefono`: Número de teléfono (opcional)
- `email`: Correo electrónico (opcional)
- `direccion`: Dirección completa (opcional)
- `fecha_registro`: Fecha de registro automática

#### Tabla: `vehiculos`
```sql
CREATE TABLE vehiculos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    matricula TEXT UNIQUE NOT NULL,
    marca TEXT NOT NULL,
    modelo TEXT NOT NULL,
    anio INTEGER,
    color TEXT,
    propietario_id INTEGER NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (propietario_id) REFERENCES propietarios(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);
```

**Campos**:
- `id`: Identificador único autoincremental
- `matricula`: Número de matrícula único (clave única)
- `marca`: Marca del vehículo
- `modelo`: Modelo del vehículo
- `anio`: Año de fabricación (opcional)
- `color`: Color del vehículo (opcional)
- `propietario_id`: Clave foránea al propietario
- `fecha_registro`: Fecha de registro automática

#### Tabla: `detecciones`
```sql
CREATE TABLE detecciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehiculo_id INTEGER NOT NULL,
    fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ubicacion TEXT,
    confianza REAL,
    imagen_path TEXT,
    FOREIGN KEY (vehiculo_id) REFERENCES vehiculos(id)
        ON DELETE CASCADE
);
```

**Campos**:
- `id`: Identificador único autoincremental
- `vehiculo_id`: Clave foránea al vehículo
- `fecha_deteccion`: Fecha y hora de la detección
- `ubicacion`: Ubicación geográfica o descripción
- `confianza`: Nivel de confianza de la detección (0-1)
- `imagen_path`: Ruta a la imagen de la detección

### Relaciones

```
propietarios (1) ──── (N) vehiculos
                         │
                         │
                         │ (1)
                         │
                         │
                         ▼
                      (N) detecciones
```

- Un propietario puede tener múltiples vehículos
- Un vehículo pertenece a un único propietario
- Un vehículo puede tener múltiples detecciones

### Índices

```sql
CREATE INDEX idx_matricula ON vehiculos(matricula);
CREATE INDEX idx_propietario_dni ON propietarios(dni);
```

Estos índices mejoran el rendimiento de las consultas más frecuentes.

---

## 🧠 Modelo de Visión Artificial

### Arquitectura: YOLOv5

**YOLO** (You Only Look Once) es una arquitectura de detección de objetos en tiempo real.

#### Características del Modelo
- **Versión**: YOLOv5 (PyTorch)
- **Tipo**: Object Detection
- **Entrada**: Imágenes RGB de tamaño configurable (default: 640x640)
- **Salida**: Bounding boxes + confianza + clase

#### Estructura de la Red

```
Input (3 x 640 x 640)
    ↓
[Backbone: CSPDarknet]
    ↓
[Neck: PANet]
    ↓
[Head: YOLO Detection Layers]
    ↓
Output: [N x (x, y, w, h, conf, class)]
```

#### Componentes

1. **Backbone (CSPDarknet)**:
   - Extracción de características
   - Múltiples escalas
   - Cross Stage Partial connections

2. **Neck (PANet)**:
   - Path Aggregation Network
   - Fusión de características multi-escala
   - Mejora detección de objetos pequeños

3. **Head (Detection)**:
   - Tres escalas de detección
   - Predicción de bounding boxes
   - Clasificación

#### Post-procesamiento

1. **Non-Maximum Suppression (NMS)**:
   ```python
   output = non_max_suppression_plate(
       pred, 
       conf_thres=0.25,  # Umbral de confianza
       iou_thres=0.45     # Umbral de IoU
   )
   ```

2. **Filtrado de confianza**: Solo se mantienen detecciones con confianza > umbral

3. **Rescalado de coordenadas**: De espacio de modelo a espacio de imagen original

#### Preprocesamiento

```python
# 1. Redimensionar a tamaño del modelo
img = cv2.resize(img0, (640, 640))

# 2. BGR -> RGB y HWC -> CHW
img = img[:, :, ::-1].transpose(2, 0, 1)

# 3. Normalizar 0-255 -> 0.0-1.0
img = img / 255.0

# 4. Convertir a tensor
img = torch.from_numpy(img).to(device)
```

### Métricas de Rendimiento

| Configuración | Precisión (mAP) | Velocidad (FPS) | Hardware |
|---------------|-----------------|-----------------|----------|
| YOLOv5s + CPU | ~0.85 | 10-15 | Intel i5 |
| YOLOv5s + GPU | ~0.85 | 60-80 | NVIDIA GTX 1660 |
| YOLOv5m + GPU | ~0.88 | 40-50 | NVIDIA GTX 1660 |
| YOLOv5l + GPU | ~0.90 | 25-30 | NVIDIA RTX 3070 |

---

## 🔌 APIs y Módulos

### API de Detección

#### Detectar en Imagen
```python
from detect_owner import LicensePlateDetector

# Crear detector
detector = LicensePlateDetector(
    weights='weights/best.pt',
    img_size=640,
    conf_thres=0.25,
    device='0'  # GPU 0
)

# Procesar imagen
results = detector.process_image(
    image_path='test.jpg',
    output_path='result.jpg',
    show=True
)

# Resultados
for result in results:
    print(f"Matrícula: {result['plate_text']}")
    print(f"Confianza: {result['detection']['confidence']:.2%}")
    if result['owner']:
        print(f"Propietario: {result['owner']['propietario']['nombre_completo']}")
```

#### Detectar en Video
```python
detector.process_video(
    video_path='traffic.mp4',
    output_path='result.mp4',
    show=True
)
```

### API de Base de Datos

#### Agregar Datos
```python
from database.vehicle_database import VehicleDatabase

with VehicleDatabase() as db:
    # Agregar propietario
    prop_id = db.agregar_propietario(
        nombre="Juan",
        apellido="Pérez",
        dni="12345678A",
        telefono="+34600123456"
    )
    
    # Agregar vehículo
    veh_id = db.agregar_vehiculo(
        matricula="1234ABC",
        marca="Toyota",
        modelo="Corolla",
        propietario_id=prop_id
    )
```

#### Consultar Datos
```python
with VehicleDatabase() as db:
    # Buscar por matrícula
    info = db.buscar_propietario_por_matricula("1234ABC")
    
    # Historial
    historial = db.obtener_historial_vehiculo("1234ABC")
    
    # Listar todos
    vehiculos = db.listar_todos_los_vehiculos()
```

---

## 🛠️ Manual de Instalación

### Instalación Completa Paso a Paso

#### 1. Requisitos Previos

**Sistema Operativo**:
- Ubuntu 18.04+ / Debian 10+
- Windows 10/11
- macOS 10.15+

**Software**:
```bash
# Python 3.7+
python --version

# Git
git --version

# CUDA (opcional, para GPU)
nvcc --version
```

#### 2. Clonar Repositorio

```bash
git clone https://github.com/zeusees/License-Plate-Detector.git
cd License-Plate-Detector
```

#### 3. Crear Entorno Virtual

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

#### 4. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

**Si tiene GPU NVIDIA con CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Si solo usa CPU**:
```bash
pip install torch torchvision torchaudio
```

#### 5. Verificar Instalación

```bash
# Test de importaciones
python -c "import torch; import cv2; print('✅ OK')"

# Test del sistema
python detect_owner.py --help
```

#### 6. Descargar Pesos del Modelo

```bash
cd weights
bash download_weights.sh
cd ..
```

O manualmente desde [releases de GitHub].

#### 7. Inicializar Base de Datos

```bash
python database/vehicle_database.py
```

#### 8. Prueba Final

```bash
# Crear imagen de prueba o usar una existente
python detect_owner.py --source imgs/test.jpg --device cpu
```

### Instalación con Docker (Opcional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "detect_owner.py"]
```

**Construir y ejecutar**:
```bash
docker build -t plate-detector .
docker run -v $(pwd)/results:/app/results plate-detector --source test.jpg
```

---

## ⚙️ Configuración Avanzada

### Variables de Entorno

```bash
# .env
DB_PATH=database/vehicles.db
MODEL_PATH=weights/best.pt
DEVICE=0
IMG_SIZE=640
CONF_THRES=0.25
IOU_THRES=0.45
```

### Configuración del Modelo

Editar configuraciones en `models/yolov5*.yaml`:

```yaml
# yolov5s.yaml (ejemplo)
nc: 1  # número de clases
depth_multiple: 0.33
width_multiple: 0.50

anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]
```

### Optimización de Rendimiento

#### Para CPU
```python
detector = LicensePlateDetector(
    device='cpu',
    img_size=416,  # Tamaño más pequeño
    conf_thres=0.4  # Mayor umbral
)
```

#### Para GPU
```python
detector = LicensePlateDetector(
    device='0',  # GPU 0
    img_size=640,
    conf_thres=0.25
)
```

### Logging Avanzado

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log'),
        logging.StreamHandler()
    ]
)
```

---

## 👨‍💻 Desarrollo y Contribución

### Estructura de Directorios

```
License-Plate-Detector/
├── database/
│   ├── vehicle_database.py    # Gestión de BD
│   └── vehicles.db            # Base de datos SQLite
├── models/
│   ├── common.py              # Bloques de la red
│   ├── yolo.py                # Arquitectura YOLOv5
│   └── *.yaml                 # Configuraciones
├── utils/
│   ├── datasets.py            # Carga de datos
│   ├── general.py             # Funciones generales
│   └── ...
├── weights/
│   └── best.pt                # Modelo entrenado
├── docs/
│   ├── MANUAL_USUARIO.md
│   └── DOCUMENTACION_TECNICA.md
├── detect_owner.py            # Script principal
├── train.py                   # Entrenamiento
├── test.py                    # Evaluación
└── requirements.txt           # Dependencias
```

### Flujo de Desarrollo

1. **Fork** del repositorio
2. **Clone** tu fork
3. **Crear rama** para tu feature
4. **Desarrollar** y **testear**
5. **Commit** con mensajes descriptivos
6. **Push** a tu fork
7. **Pull Request** al repositorio principal

### Testing

```bash
# Test unitarios
python -m pytest tests/

# Test de integración
python test.py --data data/test.yaml --weights weights/best.pt
```

### Estándares de Código

- **PEP 8** para código Python
- **Type hints** cuando sea posible
- **Docstrings** para todas las funciones públicas
- **Comentarios** para lógica compleja

### Contribuir

1. Asegúrate de que tu código pasa los tests
2. Documenta nuevas funcionalidades
3. Actualiza el README si es necesario
4. Sigue las guías de estilo del proyecto

---

## 📊 Especificaciones Técnicas

### Requisitos Mínimos

| Componente | Especificación |
|------------|----------------|
| CPU | Intel Core i5 / AMD Ryzen 5 |
| RAM | 8 GB |
| Almacenamiento | 5 GB |
| GPU (opcional) | NVIDIA GTX 1050 / AMD RX 560 |
| SO | Windows 10 / Ubuntu 18.04 / macOS 10.15 |

### Requisitos Recomendados

| Componente | Especificación |
|------------|----------------|
| CPU | Intel Core i7 / AMD Ryzen 7 |
| RAM | 16 GB |
| Almacenamiento | 10 GB SSD |
| GPU | NVIDIA RTX 2060+ / AMD RX 5700+ |
| SO | Windows 11 / Ubuntu 22.04 / macOS 12+ |

### Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| torch | ≥1.7.0 | Framework de deep learning |
| torchvision | ≥0.8.0 | Utilidades de visión artificial |
| opencv-python | ≥4.5.0 | Procesamiento de imágenes |
| numpy | ≥1.19.0 | Computación numérica |
| Pillow | ≥8.0.0 | Manejo de imágenes |
| PyYAML | ≥5.3 | Configuración |
| tqdm | ≥4.50.0 | Barras de progreso |

---

## 🔐 Seguridad y Privacidad

### Recomendaciones

1. **Datos Sensibles**: La base de datos contiene información personal
2. **Encriptación**: Considerar encriptar la base de datos en producción
3. **Acceso**: Implementar control de acceso basado en roles
4. **Logs**: No registrar información sensible en logs
5. **GDPR**: Cumplir con regulaciones de protección de datos

### Backup

```bash
# Backup manual
cp database/vehicles.db backups/vehicles_$(date +%Y%m%d).db

# Backup automático (cron)
0 2 * * * cp /path/to/database/vehicles.db /path/to/backups/vehicles_$(date +\%Y\%m\%d).db
```

---

## 📞 Soporte Técnico

Para soporte técnico avanzado:
- **Issues**: GitHub Issues del repositorio
- **Email**: Consultar README principal
- **Documentación**: Esta documentación técnica

---

**Última actualización:** Noviembre 2025  
**Versión:** 1.0  
**Autores:** Equipo de desarrollo
