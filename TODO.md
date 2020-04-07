## TODO


* Mandarfuncion bin -> uint a wrapper
* Función para conversión de bin -> int con complemento a 2
* Implementaciónde candidato multivariado
* Implementación de candidato Float
* Implementación de StopCondition::ERROR_MARGIN
* Función arbitraria de reproducción?
* Boundaries
* struct Range (Para limitar y dar estructura a los parámetros)
* Módulos
* Error Handling
* Wrapper para Parámetro con el fin de delimitar positivos, negativos y bits
  decimales

## Changelog

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
* fn reproducir_elitista(L,n):
    -Toma una lista de candidatos L con long. L.len()
    -n: Número de candidatos que tiene que regresar
* Cambiar parámetros en la fn mutate(OptimizeType)
* Modificar sort() de acuerdo a si es MIN o MAX
