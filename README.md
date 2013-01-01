kaggle-traveling-santa
======================

Some tools written in C++/CUDA to create solutions for the Traveling Santa Problem (a slight variation on the Traveling Salesman problem) as presented by Kaggle.
[View Taveling Santa Problem on Kaggle Website](http://www.kaggle.com/c/traveling-santa-problem)

##Compiling
* You will need to have gcc and nvcc installed (You're on your own there)
* Extract files, move to directory, and type `make`

##Usage
All programs output the new path to standard out.
```
nearest_neighbor [coordinate_file]
refine_path [coordinate_file] [path_file]
2_opt [coordinate_file] [path_file]
verify [coordinate_file] [path_file]
```

##Typical Workflow

Mix and repeat steps as desired.

1. Compute an initial set of paths using nearest neighbor algorithm:

    ```
    $ ./bin/nearest_neighbor santa_cities.txt > output.txt
    ```

2. Run custom path optimization to find and swap bad nodes:

    ```
    ./bin/refine_path santa_cities.txt output.txt > output_refined.txt
    ```

3. Run 2_opt optimization to find easy swaps:

    ```
    ./bin/2_opt santa_cities.txt output_refined.txt > output_2_opt.txt
    ```

4. Verify solution is valid, and compute score:

    ```
    ./bin/verify santat_cities.txt output_2_opt.txt
    ```


##Notes
The initial nearest neighbor program chooses a random starting point in the data.  Given the requirements not to share any edges with the other path, and that the algorithm is deterministic, there is a small chance it will end up running out of valid paths before it uses every node.  You can check for this by running:
```
wc -l output.txt
tail output.txt
```

Verify there are the correct number of lines and that the output from `tail` contains only nodes and no error messages.




