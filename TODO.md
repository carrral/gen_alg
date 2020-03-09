## TODO

* Función DEBUG
* Error Handling
* Módulos
* Modificar sort() de acuerdo a si es MIN o MAX
* fn diagnostics(CandidateList) -> (MaxFitness, AvgFitness)
* Boundaries

## Changelog

* Enum para tipo de optimización (MAX/MIN)
* Parámetro de StopCondition::{CICLES,ERROR}
* Verifición de funcionamiento de mutate (No parece mutar como esperado): Se queda
  en plateau en un valor arbitrario
* fn reproducir_elitista(L,n):
    -Toma una lista de candidatos L con long. L.len()
    -n: Número de candidatos que tiene que regresar
* Cambiar parámetros en la fn mutate(OptimizeType)
