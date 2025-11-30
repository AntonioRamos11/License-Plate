"""
Script de demostración completa del sistema de detección de matrículas.
Ejecute este script para ver todas las capacidades del sistema.
"""

import os
import sys

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║     🚗🔍 SISTEMA DE DETECCIÓN DE MATRÍCULAS                           ║
║         Con Identificación de Propietarios                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

print("Este script demostrará las capacidades completas del sistema.\n")
print("=" * 76)


# Paso 1: Verificar instalación
print("\n📋 PASO 1: Verificando instalación...")
print("-" * 76)

try:
    import torch
    import cv2
    import numpy as np
    print("✅ PyTorch versión:", torch.__version__)
    print("✅ OpenCV versión:", cv2.__version__)
    print("✅ NumPy versión:", np.__version__)
    print("✅ CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   GPU:", torch.cuda.get_device_name(0))
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("\n💡 Ejecute: pip install -r requirements.txt")
    sys.exit(1)


# Paso 2: Verificar base de datos
print("\n📋 PASO 2: Verificando base de datos...")
print("-" * 76)

try:
    from database.vehicle_database import VehicleDatabase
    
    db = VehicleDatabase()
    vehiculos = db.listar_todos_los_vehiculos()
    
    if len(vehiculos) == 0:
        print("⚠️  La base de datos está vacía.")
        print("💡 Ejecute: python populate_database.py")
        respuesta = input("\n¿Desea poblar la base de datos ahora? (s/n): ")
        
        if respuesta.lower() == 's':
            db.close()
            print("\n🔄 Poblando base de datos...")
            import populate_database
            populate_database.poblar_base_datos()
            
            # Reconectar
            db = VehicleDatabase()
            vehiculos = db.listar_todos_los_vehiculos()
    
    print(f"✅ Base de datos conectada: {len(vehiculos)} vehículos registrados")
    
    # Mostrar algunos vehículos
    if vehiculos:
        print("\n📋 Primeros vehículos registrados:")
        for i, v in enumerate(vehiculos[:3], 1):
            print(f"   {i}. {v['matricula']} - {v['marca']} {v['modelo']} - {v['propietario']}")
        if len(vehiculos) > 3:
            print(f"   ... y {len(vehiculos) - 3} más")
    
    db.close()
    
except Exception as e:
    print(f"❌ Error con la base de datos: {e}")
    sys.exit(1)


# Paso 3: Verificar modelo
print("\n📋 PASO 3: Verificando modelo YOLOv5...")
print("-" * 76)

weights_path = "weights/best.pt"
if os.path.exists(weights_path):
    print(f"✅ Modelo encontrado: {weights_path}")
    file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
    print(f"   Tamaño: {file_size:.2f} MB")
else:
    print(f"❌ Modelo no encontrado: {weights_path}")
    print("💡 Descargue el modelo ejecutando:")
    print("   cd weights && bash download_weights.sh")
    sys.exit(1)


# Paso 4: Test de detección
print("\n📋 PASO 4: Probando sistema de detección...")
print("-" * 76)

try:
    from detect_owner import LicensePlateDetector
    
    print("🔄 Inicializando detector...")
    detector = LicensePlateDetector(
        weights=weights_path,
        img_size=640,
        conf_thres=0.25,
        device='cpu',  # Usar CPU para demo
        db_path='database/vehicles.db'
    )
    print("✅ Detector inicializado correctamente")
    
except Exception as e:
    print(f"❌ Error al inicializar detector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Paso 5: Demo interactiva
print("\n📋 PASO 5: Demo interactiva")
print("-" * 76)
print("\n🎯 El sistema está listo para detectar matrículas!")
print("\nOpciones de uso:")
print("  1. Detectar en imagen:")
print("     python detect_owner.py --source imagen.jpg")
print("\n  2. Detectar en video:")
print("     python detect_owner.py --source video.mp4 --output resultado.mp4")
print("\n  3. Ajustar configuración:")
print("     python detect_owner.py --source imagen.jpg --conf-thres 0.5 --device 0")


# Resumen final
print("\n" + "=" * 76)
print("✅ SISTEMA COMPLETAMENTE FUNCIONAL")
print("=" * 76)

print("""
📚 Documentación disponible:
   - Manual de Usuario: docs/MANUAL_USUARIO.md
   - Documentación Técnica: docs/DOCUMENTACION_TECNICA.md
   - Guía Rápida: QUICKSTART.md

🚀 Comandos útiles:
   - Poblar BD: python populate_database.py
   - Detectar: python detect_owner.py --source imagen.jpg
   - Ayuda: python detect_owner.py --help

💡 Para comenzar, ejecute:
   python detect_owner.py --source imgs/res.jpg

""")

print("¡Gracias por usar el Sistema de Detección de Matrículas! 🚗🔍\n")
