#!/usr/bin/env python3
"""Script para agregar la matrícula AH712X a la base de datos."""

from database.vehicle_database import VehicleDatabase

def main():
    # Conectar a la base de datos
    db = VehicleDatabase('database/vehicles.db')
    
    # Agregar propietario para AH712X
    propietario_id = db.agregar_propietario(
        nombre="Carlos",
        apellido="Mendoza",
        dni="45678912C",
        telefono="+34 611 222 333",
        email="carlos.mendoza@email.com",
        direccion="Calle Principal 456, Valencia"
    )
    
    if propietario_id:
        # Agregar vehículo con matrícula AH712X
        db.agregar_vehiculo(
            matricula="AH712X",
            marca="Renault",
            modelo="Megane",
            anio=2018,
            color="Azul",
            propietario_id=propietario_id
        )
        print(f"\n✅ Matrícula AH712X agregada correctamente")
    
    # Verificar
    resultado = db.buscar_propietario_por_matricula("AH712X")
    if resultado:
        print(f"\n📋 Verificación:")
        print(f"  Matrícula: {resultado['vehiculo']['matricula']}")
        print(f"  Propietario: {resultado['propietario']['nombre_completo']}")
        print(f"  Vehículo: {resultado['vehiculo']['marca']} {resultado['vehiculo']['modelo']}")
    
    db.close()

if __name__ == "__main__":
    main()
