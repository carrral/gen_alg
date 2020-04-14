## TODO

* Nota: La mutación de MultivariedFloatCandidate tiene que llevarse a cabo en
  los dígitos más significativos.

* Ajustar métodos que puedan regresar Result (a nivel impl)

* Mejorar método .to_string() de MultivariedFloatCandidate
* Wrapper para Parámetro con el fin de delimitar positivos, negativos y bits
  decimales
* Implementación de StopCondition::ERROR_MARGIN
* Boundaries
* struct Range (Para limitar y dar estructura a los parámetros)
* Módulos
* Error Handling
* Mandarfuncion bin -> uint a wrapper

## Changelog

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
* fn reproducir_elitista(L,n):
    -Toma una lista de candidatos L con long. L.len()
    -n: Número de candidatos que tiene que regresar
* Cambiar parámetros en la fn mutate(OptimizeType)
* Modificar sort() de acuerdo a si es MIN o MAX
