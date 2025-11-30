# 🚀 Guía de Inicio Rápido

## Configuración Inicial del Sistema de Detección de Matrículas

Esta guía le ayudará a poner en marcha el sistema en menos de 10 minutos.

---

## ⚡ Pasos Rápidos

### 1. Clonar el Repositorio
```bash
git clone https://github.com/zeusees/License-Plate-Detector.git
cd License-Plate-Detector
```

### 2. Instalar Python y Dependencias

**Verificar Python (requiere 3.7+):**
```bash
python --version
```

**Crear entorno virtual:**
```bash
python -m venv venv

# Activar en Linux/macOS:
source venv/bin/activate

# Activar en Windows:
venv\Scripts\activate
```

**Instalar dependencias:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Descargar Modelo

```bash
cd weights
bash download_weights.sh  # Linux/macOS
# O descargar manualmente desde releases de GitHub
cd ..
```

### 4. Inicializar Base de Datos con Datos de Ejemplo

```bash
# Esto crea la base de datos y agrega datos de ejemplo
python populate_database.py
```

### 5. ¡Ejecutar Primera Detección!

```bash
# Si tiene una imagen de prueba:
python detect_owner.py --source ruta/a/imagen.jpg

# Usar imagen de ejemplo (si existe):
python detect_owner.py --source imgs/res.jpg
```

---

## 🎯 Verificación de Instalación

### Test 1: Importaciones
```bash
python -c "import torch; import cv2; import numpy; print('✅ Todas las importaciones OK')"
```

### Test 2: Base de Datos
```bash
python -c "from database.vehicle_database import VehicleDatabase; db = VehicleDatabase(); print('✅ Base de datos OK'); db.close()"
```

### Test 3: Ayuda del Sistema
```bash
python detect_owner.py --help
```

Si todos los tests pasan, ¡está listo para usar el sistema! 🎉

---

## 🔧 Configuración Recomendada

### Para CPU Solamente
```bash
python detect_owner.py --source imagen.jpg --device cpu --img-size 416
```

### Para GPU NVIDIA
```bash
# Verificar CUDA
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

# Ejecutar con GPU
python detect_owner.py --source imagen.jpg --device 0
```

---

## 📚 Próximos Pasos

1. **Leer el Manual de Usuario**: `docs/MANUAL_USUARIO.md`
2. **Agregar tus propios datos**: Usar la API de base de datos
3. **Personalizar detección**: Ajustar umbrales y parámetros
4. **Explorar documentación técnica**: `docs/DOCUMENTACION_TECNICA.md`

---

## ❓ Problemas Comunes

### Problema: "torch not found"
```bash
pip install torch torchvision torchaudio
```

### Problema: "cv2 not found"
```bash
pip install opencv-python
```

### Problema: "No se encuentra el modelo"
- Descargar manualmente `best.pt` y colocarlo en `weights/`

### Problema: "Permission denied en scripts"
```bash
chmod +x weights/download_weights.sh
```

---

## 🆘 Soporte

- 📖 **Documentación completa**: Carpeta `docs/`
- 🐛 **Reportar bugs**: GitHub Issues
- 💬 **Preguntas**: Ver FAQ en Manual de Usuario

---

¡Feliz detección! 🚗🔍
