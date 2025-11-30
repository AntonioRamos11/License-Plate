# 🗄️ Sistema de Base de Datos

Este directorio contiene el módulo de gestión de base de datos para el sistema de detección de matrículas.

## 📁 Contenido

- `vehicle_database.py` - Módulo principal de gestión de base de datos
- `vehicles.db` - Base de datos SQLite (creada automáticamente)

## 🚀 Uso Rápido

### Crear e Inicializar Base de Datos

```python
from database.vehicle_database import VehicleDatabase

# Crear instancia (crea BD automáticamente)
db = VehicleDatabase()

# O usar como context manager (recomendado)
with VehicleDatabase() as db:
    # operaciones...
    pass
```

### Agregar Propietario

```python
with VehicleDatabase() as db:
    propietario_id = db.agregar_propietario(
        nombre="Juan",
        apellido="Pérez",
        dni="12345678A",
        telefono="+34600123456",
        email="juan@email.com",
        direccion="Calle Mayor 1"
    )
```

### Agregar Vehículo

```python
with VehicleDatabase() as db:
    vehiculo_id = db.agregar_vehiculo(
        matricula="1234ABC",
        marca="Toyota",
        modelo="Corolla",
        anio=2020,
        color="Blanco",
        propietario_id=1
    )
```

### Buscar Propietario por Matrícula

```python
with VehicleDatabase() as db:
    info = db.buscar_propietario_por_matricula("1234ABC")
    
    if info:
        print(f"Propietario: {info['propietario']['nombre_completo']}")
        print(f"Vehículo: {info['vehiculo']['marca']} {info['vehiculo']['modelo']}")
```

### Registrar Detección

```python
with VehicleDatabase() as db:
    deteccion_id = db.registrar_deteccion(
        vehiculo_id=1,
        ubicacion="Calle Principal",
        confianza=0.95,
        imagen_path="detecciones/img001.jpg"
    )
```

### Ver Historial

```python
with VehicleDatabase() as db:
    historial = db.obtener_historial_vehiculo("1234ABC", limit=10)
    
    for det in historial:
        print(f"Fecha: {det['fecha']}")
        print(f"Ubicación: {det['ubicacion']}")
        print(f"Confianza: {det['confianza']:.2%}")
```

## 📊 Esquema de la Base de Datos

### Tabla: propietarios
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

### Tabla: vehiculos
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
);
```

### Tabla: detecciones
```sql
CREATE TABLE detecciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehiculo_id INTEGER NOT NULL,
    fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ubicacion TEXT,
    confianza REAL,
    imagen_path TEXT,
    FOREIGN KEY (vehiculo_id) REFERENCES vehiculos(id)
);
```

## 🔧 API Completa

### Clase VehicleDatabase

#### Métodos de Propietarios
- `agregar_propietario(nombre, apellido, dni, telefono, email, direccion)`
- `actualizar_propietario(propietario_id, **kwargs)`

#### Métodos de Vehículos
- `agregar_vehiculo(matricula, marca, modelo, propietario_id, anio, color)`
- `buscar_propietario_por_matricula(matricula)`
- `listar_todos_los_vehiculos()`
- `eliminar_vehiculo(matricula)`

#### Métodos de Detecciones
- `registrar_deteccion(vehiculo_id, ubicacion, confianza, imagen_path)`
- `obtener_historial_vehiculo(matricula, limit)`

## 🔒 Seguridad

- ✅ Claves foráneas habilitadas
- ✅ Constraints UNIQUE en DNI y matrícula
- ✅ Cascada en eliminaciones
- ✅ Validación de datos
- ✅ Manejo de errores robusto

## 📝 Notas

- La base de datos se crea automáticamente en la primera ejecución
- Se crean índices automáticamente para optimizar consultas
- Las fechas se registran automáticamente
- Soporte para context manager (with statement)

## 🛠️ Mantenimiento

### Backup de Base de Datos
```bash
cp database/vehicles.db backup/vehicles_$(date +%Y%m%d).db
```

### Ver contenido (usando sqlite3)
```bash
sqlite3 database/vehicles.db
# Dentro de sqlite3:
.tables
SELECT * FROM propietarios;
SELECT * FROM vehiculos;
SELECT * FROM detecciones;
```

### Restaurar desde Backup
```bash
cp backup/vehicles_20231130.db database/vehicles.db
```

## 📚 Más Información

Ver la documentación completa en:
- `docs/DOCUMENTACION_TECNICA.md`
- `docs/MANUAL_USUARIO.md`
