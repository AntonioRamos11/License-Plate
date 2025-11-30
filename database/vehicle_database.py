"""
Sistema de Base de Datos para Gestión de Vehículos y Propietarios
Este módulo proporciona funcionalidades para gestionar la base de datos
de vehículos y sus propietarios.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path


class VehicleDatabase:
    """Clase para gestionar la base de datos de vehículos y propietarios."""
    
    def __init__(self, db_path='database/vehicles.db'):
        """
        Inicializa la conexión a la base de datos.
        
        Args:
            db_path (str): Ruta al archivo de base de datos SQLite
        """
        self.db_path = db_path
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establece conexión con la base de datos."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            # Habilitar claves foráneas
            self.cursor.execute("PRAGMA foreign_keys = ON")
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error al conectar con la base de datos: {e}")
            raise
    
    def _create_tables(self):
        """Crea las tablas necesarias en la base de datos."""
        try:
            # Tabla de propietarios
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS propietarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL,
                    apellido TEXT NOT NULL,
                    dni TEXT UNIQUE NOT NULL,
                    telefono TEXT,
                    email TEXT,
                    direccion TEXT,
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de vehículos
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehiculos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matricula TEXT UNIQUE NOT NULL,
                    marca TEXT NOT NULL,
                    modelo TEXT NOT NULL,
                    anio INTEGER,
                    color TEXT,
                    propietario_id INTEGER NOT NULL,
                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (propietario_id) REFERENCES propietarios(id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE
                )
            """)
            
            # Tabla de detecciones (historial)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS detecciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehiculo_id INTEGER NOT NULL,
                    fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ubicacion TEXT,
                    confianza REAL,
                    imagen_path TEXT,
                    FOREIGN KEY (vehiculo_id) REFERENCES vehiculos(id)
                        ON DELETE CASCADE
                )
            """)
            
            # Índices para mejorar el rendimiento
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matricula 
                ON vehiculos(matricula)
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_propietario_dni 
                ON propietarios(dni)
            """)
            
            self.conn.commit()
            print("✅ Tablas de base de datos creadas exitosamente")
            
        except sqlite3.Error as e:
            print(f"Error al crear las tablas: {e}")
            raise
    
    def agregar_propietario(self, nombre, apellido, dni, telefono=None, 
                           email=None, direccion=None):
        """
        Agrega un nuevo propietario a la base de datos.
        
        Args:
            nombre (str): Nombre del propietario
            apellido (str): Apellido del propietario
            dni (str): DNI único del propietario
            telefono (str, optional): Número de teléfono
            email (str, optional): Correo electrónico
            direccion (str, optional): Dirección
            
        Returns:
            int: ID del propietario insertado, o None si hay error
        """
        try:
            self.cursor.execute("""
                INSERT INTO propietarios (nombre, apellido, dni, telefono, email, direccion)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (nombre, apellido, dni, telefono, email, direccion))
            self.conn.commit()
            print(f"✅ Propietario {nombre} {apellido} agregado exitosamente")
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"⚠️ Error: El DNI {dni} ya existe en la base de datos")
            return None
        except sqlite3.Error as e:
            print(f"❌ Error al agregar propietario: {e}")
            return None
    
    def agregar_vehiculo(self, matricula, marca, modelo, propietario_id, 
                        anio=None, color=None):
        """
        Agrega un nuevo vehículo a la base de datos.
        
        Args:
            matricula (str): Número de matrícula único
            marca (str): Marca del vehículo
            modelo (str): Modelo del vehículo
            propietario_id (int): ID del propietario
            anio (int, optional): Año del vehículo
            color (str, optional): Color del vehículo
            
        Returns:
            int: ID del vehículo insertado, o None si hay error
        """
        try:
            self.cursor.execute("""
                INSERT INTO vehiculos (matricula, marca, modelo, anio, color, propietario_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (matricula.upper(), marca, modelo, anio, color, propietario_id))
            self.conn.commit()
            print(f"✅ Vehículo con matrícula {matricula} agregado exitosamente")
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"⚠️ Error: La matrícula {matricula} ya existe en la base de datos")
            return None
        except sqlite3.Error as e:
            print(f"❌ Error al agregar vehículo: {e}")
            return None
    
    def buscar_propietario_por_matricula(self, matricula):
        """
        Busca el propietario de un vehículo por su matrícula.
        
        Args:
            matricula (str): Número de matrícula a buscar
            
        Returns:
            dict: Información del propietario y vehículo, o None si no existe
        """
        try:
            self.cursor.execute("""
                SELECT 
                    p.id, p.nombre, p.apellido, p.dni, p.telefono, p.email, p.direccion,
                    v.id as vehiculo_id, v.matricula, v.marca, v.modelo, v.anio, v.color
                FROM propietarios p
                INNER JOIN vehiculos v ON p.id = v.propietario_id
                WHERE v.matricula = ?
            """, (matricula.upper(),))
            
            resultado = self.cursor.fetchone()
            
            if resultado:
                return {
                    'propietario': {
                        'id': resultado[0],
                        'nombre': resultado[1],
                        'apellido': resultado[2],
                        'nombre_completo': f"{resultado[1]} {resultado[2]}",
                        'dni': resultado[3],
                        'telefono': resultado[4],
                        'email': resultado[5],
                        'direccion': resultado[6]
                    },
                    'vehiculo': {
                        'id': resultado[7],
                        'matricula': resultado[8],
                        'marca': resultado[9],
                        'modelo': resultado[10],
                        'anio': resultado[11],
                        'color': resultado[12]
                    }
                }
            return None
            
        except sqlite3.Error as e:
            print(f"❌ Error al buscar propietario: {e}")
            return None
    
    def registrar_deteccion(self, vehiculo_id, ubicacion=None, 
                           confianza=None, imagen_path=None):
        """
        Registra una detección de vehículo en el historial.
        
        Args:
            vehiculo_id (int): ID del vehículo detectado
            ubicacion (str, optional): Ubicación de la detección
            confianza (float, optional): Nivel de confianza de la detección
            imagen_path (str, optional): Ruta a la imagen de la detección
            
        Returns:
            int: ID de la detección registrada, o None si hay error
        """
        try:
            self.cursor.execute("""
                INSERT INTO detecciones (vehiculo_id, ubicacion, confianza, imagen_path)
                VALUES (?, ?, ?, ?)
            """, (vehiculo_id, ubicacion, confianza, imagen_path))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"❌ Error al registrar detección: {e}")
            return None
    
    def obtener_historial_vehiculo(self, matricula, limit=10):
        """
        Obtiene el historial de detecciones de un vehículo.
        
        Args:
            matricula (str): Matrícula del vehículo
            limit (int): Número máximo de registros a retornar
            
        Returns:
            list: Lista de detecciones
        """
        try:
            self.cursor.execute("""
                SELECT d.id, d.fecha_deteccion, d.ubicacion, d.confianza, d.imagen_path
                FROM detecciones d
                INNER JOIN vehiculos v ON d.vehiculo_id = v.id
                WHERE v.matricula = ?
                ORDER BY d.fecha_deteccion DESC
                LIMIT ?
            """, (matricula.upper(), limit))
            
            resultados = self.cursor.fetchall()
            return [{
                'id': r[0],
                'fecha': r[1],
                'ubicacion': r[2],
                'confianza': r[3],
                'imagen': r[4]
            } for r in resultados]
            
        except sqlite3.Error as e:
            print(f"❌ Error al obtener historial: {e}")
            return []
    
    def listar_todos_los_vehiculos(self):
        """
        Lista todos los vehículos registrados.
        
        Returns:
            list: Lista de todos los vehículos con información de propietarios
        """
        try:
            self.cursor.execute("""
                SELECT v.matricula, v.marca, v.modelo, v.anio, v.color,
                       p.nombre, p.apellido, p.dni
                FROM vehiculos v
                INNER JOIN propietarios p ON v.propietario_id = p.id
                ORDER BY v.matricula
            """)
            
            resultados = self.cursor.fetchall()
            return [{
                'matricula': r[0],
                'marca': r[1],
                'modelo': r[2],
                'anio': r[3],
                'color': r[4],
                'propietario': f"{r[5]} {r[6]}",
                'dni_propietario': r[7]
            } for r in resultados]
            
        except sqlite3.Error as e:
            print(f"❌ Error al listar vehículos: {e}")
            return []
    
    def actualizar_propietario(self, propietario_id, **kwargs):
        """
        Actualiza la información de un propietario.
        
        Args:
            propietario_id (int): ID del propietario
            **kwargs: Campos a actualizar
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario
        """
        campos_validos = ['nombre', 'apellido', 'telefono', 'email', 'direccion']
        campos_actualizar = {k: v for k, v in kwargs.items() if k in campos_validos}
        
        if not campos_actualizar:
            print("⚠️ No hay campos válidos para actualizar")
            return False
        
        try:
            set_clause = ", ".join([f"{k} = ?" for k in campos_actualizar.keys()])
            valores = list(campos_actualizar.values()) + [propietario_id]
            
            self.cursor.execute(f"""
                UPDATE propietarios
                SET {set_clause}
                WHERE id = ?
            """, valores)
            
            self.conn.commit()
            print(f"✅ Propietario {propietario_id} actualizado exitosamente")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Error al actualizar propietario: {e}")
            return False
    
    def eliminar_vehiculo(self, matricula):
        """
        Elimina un vehículo de la base de datos.
        
        Args:
            matricula (str): Matrícula del vehículo a eliminar
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        try:
            self.cursor.execute("""
                DELETE FROM vehiculos WHERE matricula = ?
            """, (matricula.upper(),))
            
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                print(f"✅ Vehículo con matrícula {matricula} eliminado exitosamente")
                return True
            else:
                print(f"⚠️ No se encontró vehículo con matrícula {matricula}")
                return False
                
        except sqlite3.Error as e:
            print(f"❌ Error al eliminar vehículo: {e}")
            return False
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            print("🔒 Conexión a la base de datos cerrada")
    
    def __enter__(self):
        """Soporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cierra la conexión al salir del context manager."""
        self.close()


# Script de demostración
if __name__ == "__main__":
    print("=" * 60)
    print("  Sistema de Base de Datos para Detección de Matrículas")
    print("=" * 60)
    
    # Crear instancia de la base de datos
    with VehicleDatabase() as db:
        # Agregar propietarios de ejemplo
        print("\n📝 Agregando propietarios de ejemplo...")
        prop1_id = db.agregar_propietario(
            nombre="Juan",
            apellido="Pérez",
            dni="12345678A",
            telefono="+34 600 123 456",
            email="juan.perez@email.com",
            direccion="Calle Mayor 123, Madrid"
        )
        
        prop2_id = db.agregar_propietario(
            nombre="María",
            apellido="García",
            dni="87654321B",
            telefono="+34 600 654 321",
            email="maria.garcia@email.com",
            direccion="Avenida Libertad 45, Barcelona"
        )
        
        # Agregar vehículos de ejemplo
        print("\n🚗 Agregando vehículos de ejemplo...")
        if prop1_id:
            db.agregar_vehiculo(
                matricula="1234ABC",
                marca="Toyota",
                modelo="Corolla",
                anio=2020,
                color="Blanco",
                propietario_id=prop1_id
            )
        
        if prop2_id:
            db.agregar_vehiculo(
                matricula="5678DEF",
                marca="Honda",
                modelo="Civic",
                anio=2019,
                color="Negro",
                propietario_id=prop2_id
            )
        
        # Buscar propietario por matrícula
        print("\n🔍 Buscando propietario de la matrícula 1234ABC...")
        resultado = db.buscar_propietario_por_matricula("1234ABC")
        
        if resultado:
            print("\n✅ Vehículo encontrado:")
            print(f"  Matrícula: {resultado['vehiculo']['matricula']}")
            print(f"  Vehículo: {resultado['vehiculo']['marca']} {resultado['vehiculo']['modelo']}")
            print(f"  Propietario: {resultado['propietario']['nombre_completo']}")
            print(f"  DNI: {resultado['propietario']['dni']}")
            print(f"  Teléfono: {resultado['propietario']['telefono']}")
            print(f"  Email: {resultado['propietario']['email']}")
        
        # Listar todos los vehículos
        print("\n📋 Listado de todos los vehículos registrados:")
        vehiculos = db.listar_todos_los_vehiculos()
        for v in vehiculos:
            print(f"  {v['matricula']} - {v['marca']} {v['modelo']} - Propietario: {v['propietario']}")
        
        print("\n" + "=" * 60)
        print("  Demostración completada exitosamente")
        print("=" * 60)
