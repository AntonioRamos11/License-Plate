"""
Script de ejemplo para poblar la base de datos con datos de prueba.
Ejecute este script después de crear la base de datos para tener datos de ejemplo.
"""

from database.vehicle_database import VehicleDatabase


def poblar_base_datos():
    """Puebla la base de datos con datos de ejemplo."""
    
    print("=" * 70)
    print("  Poblando Base de Datos con Datos de Ejemplo")
    print("=" * 70)
    
    with VehicleDatabase() as db:
        # Lista de propietarios de ejemplo
        propietarios = [
            {
                "nombre": "Juan",
                "apellido": "Pérez García",
                "dni": "12345678A",
                "telefono": "+34 600 123 456",
                "email": "juan.perez@email.com",
                "direccion": "Calle Mayor 123, 28001 Madrid"
            },
            {
                "nombre": "María",
                "apellido": "García López",
                "dni": "87654321B",
                "telefono": "+34 600 654 321",
                "email": "maria.garcia@email.com",
                "direccion": "Avenida Libertad 45, 08001 Barcelona"
            },
            {
                "nombre": "Carlos",
                "apellido": "Rodríguez Martínez",
                "dni": "11223344C",
                "telefono": "+34 600 112 233",
                "email": "carlos.rodriguez@email.com",
                "direccion": "Plaza España 7, 41001 Sevilla"
            },
            {
                "nombre": "Ana",
                "apellido": "Martínez Sánchez",
                "dni": "44332211D",
                "telefono": "+34 600 443 322",
                "email": "ana.martinez@email.com",
                "direccion": "Calle Valencia 89, 46001 Valencia"
            },
            {
                "nombre": "Luis",
                "apellido": "Fernández Gómez",
                "dni": "55667788E",
                "telefono": "+34 600 556 677",
                "email": "luis.fernandez@email.com",
                "direccion": "Gran Vía 234, 48001 Bilbao"
            }
        ]
        
        # Lista de vehículos de ejemplo
        vehiculos = [
            {
                "matricula": "1234ABC",
                "marca": "Toyota",
                "modelo": "Corolla",
                "anio": 2020,
                "color": "Blanco",
                "propietario_idx": 0
            },
            {
                "matricula": "5678DEF",
                "marca": "Honda",
                "modelo": "Civic",
                "anio": 2019,
                "color": "Negro",
                "propietario_idx": 1
            },
            {
                "matricula": "9012GHI",
                "marca": "Ford",
                "modelo": "Focus",
                "anio": 2021,
                "color": "Azul",
                "propietario_idx": 2
            },
            {
                "matricula": "3456JKL",
                "marca": "Volkswagen",
                "modelo": "Golf",
                "anio": 2018,
                "color": "Gris",
                "propietario_idx": 3
            },
            {
                "matricula": "7890MNO",
                "marca": "Seat",
                "modelo": "León",
                "anio": 2022,
                "color": "Rojo",
                "propietario_idx": 4
            },
            {
                "matricula": "2468PQR",
                "marca": "BMW",
                "modelo": "Serie 3",
                "anio": 2020,
                "color": "Negro",
                "propietario_idx": 0
            },
            {
                "matricula": "1357STU",
                "marca": "Mercedes",
                "modelo": "Clase A",
                "anio": 2021,
                "color": "Blanco",
                "propietario_idx": 1
            },
            {
                "matricula": "9753VWX",
                "marca": "Audi",
                "modelo": "A4",
                "anio": 2019,
                "color": "Gris",
                "propietario_idx": 2
            }
        ]
        
        # Agregar propietarios
        print("\n📝 Agregando propietarios...")
        propietarios_ids = []
        for prop in propietarios:
            prop_id = db.agregar_propietario(**prop)
            if prop_id:
                propietarios_ids.append(prop_id)
                print(f"  ✅ {prop['nombre']} {prop['apellido']} - DNI: {prop['dni']}")
        
        # Agregar vehículos
        print("\n🚗 Agregando vehículos...")
        for veh in vehiculos:
            propietario_id = propietarios_ids[veh['propietario_idx']]
            veh_data = {k: v for k, v in veh.items() if k != 'propietario_idx'}
            veh_data['propietario_id'] = propietario_id
            
            veh_id = db.agregar_vehiculo(**veh_data)
            if veh_id:
                print(f"  ✅ {veh['matricula']} - {veh['marca']} {veh['modelo']} ({veh['color']})")
        
        # Mostrar resumen
        print("\n" + "=" * 70)
        print("  Resumen de la Base de Datos")
        print("=" * 70)
        
        vehiculos_lista = db.listar_todos_los_vehiculos()
        print(f"\n📊 Total de vehículos registrados: {len(vehiculos_lista)}")
        print("\n📋 Listado completo:")
        print(f"{'Matrícula':<12} {'Vehículo':<25} {'Propietario':<25} {'DNI':<12}")
        print("-" * 80)
        
        for v in vehiculos_lista:
            vehiculo = f"{v['marca']} {v['modelo']}"
            print(f"{v['matricula']:<12} {vehiculo:<25} {v['propietario']:<25} {v['dni_propietario']:<12}")
        
        print("\n" + "=" * 70)
        print("  ✅ Base de datos poblada exitosamente")
        print("=" * 70)
        print("\n💡 Ahora puede ejecutar detecciones con:")
        print("   python detect_owner.py --source imagen.jpg\n")


if __name__ == "__main__":
    poblar_base_datos()
