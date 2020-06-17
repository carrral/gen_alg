## TODO

* Niveles de verbosidad
* Ajustar métodos que puedan regresar Result (a nivel trait)
* Implementación de StopCondition::ERROR_MARGIN
* Error Handling
* Mandar función bin -> uint a wrapper
* Añadir benchmarks para optimizar velocidad.

## Changelog

* Cambiar el nombre a MultivaluedFloat
* Implementación de trait FitnessFunction
* Guardar máximos y mínimos globales dentro del algoritmo
* Corregir: la función de reproducción no elige correctamente los valores
* Bounds
* Módulos
* Ahora FitnessFunction se procesa dentro de un Box<dyn F(U) -> FitnessReturn>
* trait Function{} (wrapper)
* Implementaciónde candidato multivariado Float
* Implementación de MVFCandidateList
* Función arbitraria de reproducción? (IMPOSIBLE CON LA IMPLEMENTACIÓN ACTUAL DE
  Candidate)
* Correción de errores en StopCondition::BOUND() [Loop infinito]
* Función arbitraria de generación de cadenas de bits
* Diferentes ramas para diferentes condiciones de paro
* Función DEBUG
* struct InternalState: InternalState.meets(&StopCondition)
* fn diagnostics(CandidateList) -> (MaxFitness, AvgFitness)
* Separar implementación de definición
* Marcar código no genérico
* Enum para tipo de optimización (MAX/MIN)
* Implementar ord, eq para f64 
    NOTA: No se implmenetaron los traits, pero se cambió el método de
    ordenamiento
* Parámetro de StopCondition::{CICLES,ERROR}
* Verifición de funcionamiento de mutate (No parece mutar como esperado): Se queda
  en plateau en un valor arbitrario
* Modificar sort() de acuerdo a si es MIN o MAX
