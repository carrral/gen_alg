# Implementación de Algoritmo Genético

## Requerimientos

Para correr el algoritmo principal es necesario tener instalado `cargo`.

Para mostrar los ploteos es necesaria la instalación de `GNU Plot`.

## Parámetros

La función `basic_genetic_algorithm()` toma los siguientes parámetros:

    n:  Tamaño de la población inicial

    selected_per_round: Número de candidatos seleccionados para reproducción por ronda.

    candidates: Implementación de CandidateList. 

    fitness_fn: Función de adaptación o función a optimizar.

    mating_pr:  Probabilidad de reproducción.

    mut_pr: Probabilidad de mutación. 

    opt: Tipo de optimización (Minimización/Maximización).

    stop_cond: Condición de paro (Ciclos, Tope de fitness) 

    debug_value: Mostrar datos de debug (V/F).

    show_fitness_plot: Mostrar ploteos de fitness (V/F).

Actualmente, el algoritmo está configurado para maximizar la función 
    f(x,y) = -(x - 5)² - (y - 7)² + 5
con óptimo global en (5,7)


## Correr el programa

Dentro de la carpeta de instalación, ejecutar

    $ cargo run > results.txt
Esto almacenará los resultados en result.txt
