## TODO

* Marcar código no genérico
* Boundaries
* Módulos
* Error Handling
* fn diagnostics(CandidateList) -> (MaxFitness, AvgFitness)
* Función DEBUG
* Diferentes ramas para diferentes condiciones de paro
* struct InternalState: InternalState.meets(&StopCondition)
* struct Range (Para limitar y dar estructura a los parámetros)
* Wrapper para Parámetro con el fin de delimitar positivos, negativos y bits
  decimales

## Changelog

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
