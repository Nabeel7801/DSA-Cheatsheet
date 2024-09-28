## Sorting
<details>
  <summary>Merge Sort</summary>
    
  ### Merge Sort
  
  An efficient, stable, divide and conquer sorting algorithm.
    
  ```javascript
    function mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const left = mergeSort(arr.slice(0, mid));
        const right = mergeSort(arr.slice(mid));
        
        return merge(left, right);
    }
    
    function merge(left, right) {
        let result = [];
        let i = 0, j = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] < right[j]) {
                result.push(left[i++]);
            } else {
                result.push(right[j++]);
            }
        }
        
        return result.concat(left.slice(i)).concat(right.slice(j));
    }
  ```
</details>

<details>
  <summary>Quick Sort</summary>
    
  ### Quick Sort
  
  An algorithm to find the maximum sum of a contiguous subarray.
    
  ```javascript
    function quickSort(arr) {
        if (arr.length <= 1) return arr;
        
        const pivot = arr[arr.length - 1];
        const left = [], right = [];
        
        for (let i = 0; i < arr.length - 1; i++) {
            if (arr[i] < pivot) {
                left.push(arr[i]);
            } else {
                right.push(arr[i]);
            }
        }
        
        return [...quickSort(left), pivot, ...quickSort(right)];
    }
  ```
</details>


## Searching
<details>
  <summary>Binary Search</summary>
    
  ### Binary Search
  
  A divide and conquer algorithm to find the position of an element in a sorted array.
    
  ```javascript
   function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // Target not found
  }
  ```
</details>

<details>
  <summary>Depth First Search (DFS)</summary>
    
  ### Depth First Search (DFS)
  
  A graph traversal algorithm that explores as far as possible along each branch before backtracking.
    
  ```javascript
    const graph = [];
    const visited = new Set();
    
    function dfs(index) {
        if (visited.has(index));
        visited.add(index);
        
        graph[index].forEach((neighbor) => {
            if (!visited.has(neighbor)) {
                dfs(neighbor);
            }
        });
    }
  ```
</details>

<details>
  <summary>Breadth First Search (BFS)</summary>
    
  ### Breadth First Search (BFS)
  
  A graph traversal algorithm that explores all the neighbors at the present depth before moving on to nodes at the next depth level.
    
  ```javascript
    const graph = [];
    function bfs(index) {
        let queue = [index];
    
        let visited = new Set();
        visited.add(index);
        
        while (queue.length > 0) {
            let node = queue.shift();
            
            graph[node].forEach((neighbor) => {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            });
        }
    }
  ```
</details>


## Shortest Path
<details> 
  <summary>Dijkstra’s Algorithm</summary>

  ### Dijkstra’s Algorithm

  An algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks.

  ```javascript
    const graph = [];
    function dijkstra(start) {
        let distances = {};
        let visited = new Set();
        
        for (let node in graph) {
            distances[node] = Infinity;
        }
        distances[start] = 0;
        
        while (visited.size !== Object.keys(graph).length) {
            let closestNode = null;
            for (let node in distances) {
                if (!visited.has(node)) {
                    if (closestNode === null || distances[node] < distances[closestNode]) {
                        closestNode = node;
                    }
                }
            }
            
            visited.add(closestNode);
            
            for (let neighbor in graph[closestNode]) {
                let newDist = distances[closestNode] + graph[closestNode][neighbor];
                if (newDist < distances[neighbor]) {
                    distances[neighbor] = newDist;
                }
            }
        }
        
        return distances;
    }
  ```
</details>

<details> 
  <summary>Floyd-Warshall Algorithm</summary>

  ### Floyd-Warshall Algorithm

  A dynamic programming algorithm for finding shortest paths in a weighted graph with positive or negative edge weights.

  ```javascript
  function floydWarshall(graph) {
    let dist = [];
    const V = graph.length;
    
    for (let i = 0; i < V; i++) {
        dist[i] = [];
        for (let j = 0; j < V; j++) {
            dist[i][j] = graph[i][j];
        }
    }
    
    for (let k = 0; k < V; k++) {
        for (let i = 0; i < V; i++) {
            for (let j = 0; j < V; j++) {
                dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
    
    return dist;
  }
  ```
</details>


## Strings
<details> 
  <summary>KMP Pattern Matching (Prefix Function)</summary>

  ### KMP Pattern Matching (Prefix Function)

  Efficient string searching algorithm (Knuth-Morris-Pratt). Find the length of the longest proper prefix of the substring which is also a suffix of this substring

  ```javascript
  function kmpPrefixFunction(s) {
      const prefix = Array(s.length).fill(0);
      for (let i = 1, j = 0; i < s.length; i++) {
          while (j > 0 && s[i] !== s[j]) j = prefix[j - 1];
          if (s[i] === s[j]) j++;
          prefix[i] = j;
      }
      return prefix;
  }

  const str = "ababcab"
  kmpPrefixFunction(str);
  ```
</details>

<details>
  <summary>Z Function (Pattern Matching)</summary>
  
  ### Z Function (Pattern Matching)
  
  The Z-function for a string is an array where the value at index `i` is the length of the longest substring starting from `i` that is also a prefix of the string.
  
  ```javascript
  function zFunction(s) {
      const Z = Array(s.length).fill(0);
      let L = 0, R = 0;
      for (let i = 1; i < s.length; i++) {
          if (i <= R) Z[i] = Math.min(R - i + 1, Z[i - L]);
          while (i + Z[i] < s.length && s[Z[i]] === s[i + Z[i]]) Z[i]++;
          if (i + Z[i] - 1 > R) [L, R] = [i, i + Z[i] - 1];
      }
      return Z;
  }

  const pattern = "abc"
  const str = "ababc"
  zFunction(pattern + '#' + str);
  ```
</details>


## Sums
<details>
  <summary>Kadane’s Algorithm (Maximum Subarray Problem)</summary>
    
  ### Kadane’s Algorithm (Maximum Subarray Problem)
  
  An algorithm to find the maximum sum of a contiguous subarray.
    
  ```javascript
     function maxSubArray(arr) {
        let maxCurrent = arr[0];
        let maxGlobal = arr[0];
        
        for (let i = 1; i < arr.length; i++) {
            maxCurrent = Math.max(arr[i], maxCurrent + arr[i]);
            if (maxCurrent > maxGlobal) {
                maxGlobal = maxCurrent;
            }
        }
        
        return maxGlobal;
    }
  ```
</details>

<details> 
  <summary>Sum of Digits</summary>

  ### Sum of Digits

  Finds the sum of digits of a given number.

  ```javascript
  function sumOfDigits(n) {
      return n.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0);
  }
  ```
</details>


## Math
<details> 
  <summary>Sieve of Eratosthenes (Finding All Primes Up to N)</summary>

  ### Sieve of Eratosthenes (Finding All Primes Up to N)

  An efficient algorithm to find all primes less than or equal to n.

  ```javascript
  function sieveOfEratosthenes(n) {
    const primes = Array(n + 1).fill(true);
    primes[0] = primes[1] = false;
    
    for (let i = 2; i * i <= n; i++) {
        if (primes[i]) {
            for (let j = i * i; j <= n; j += i) {
                primes[j] = false;
            }
        }
    }
    
    return primes.map((isPrime, index) => isPrime ? index : null).filter(Boolean);
  }
  ```
</details>

<details> 
  <summary>Modular Inverse (Fermat's Little Theorem)</summary>

  ### Modular Inverse (Fermat's Little Theorem)

  Finds the modular inverse of a under modulo m when m is prime.

  ```javascript
  function modInverse(a, m) {
      return modPow(a, m - 2, m); // Using Fermat's Little Theorem
  }
  ```
</details>

<details> 
  <summary>Greatest Common Divisor (GCD)</summary>
  
  ### Greatest Common Divisor (GCD)
  
  The greatest common divisor of two numbers using Euclid's algorithm.
  
  ```javascript
  function gcd(a, b) {
      return b === 0 ? a : gcd(b, a % b);
  }
  ```
</details>

<details> 
  <summary>Least Common Multiple (LCM)</summary>
  
  ### Least Common Multiple (LCM)
  
  The least common multiple of two numbers.
  
  ```javascript
  function lcm(a, b) {
      return (a * b) / gcd(a, b);
  }
  ```
</details> 

<details> 
  <summary>nCr (Combinations)</summary>

  ### nCr (Combinations)

  Calculates the number of combinations (n choose r).

  ```javascript
  function nCr(n, r) {
      if (r > n) return 0;
      let res = 1;
      for (let i = 0; i < r; i++) {
          res *= (n - i);
          res /= (i + 1);
      }
      return res;
  }
  ```
</details>

<details> 
  <summary>Factorial</summary>

  ### Factorial

  Calculates the factorial of a number n recursively.

  ```javascript
  function factorial(n) {
      return n === 0 ? 1 : n * factorial(n - 1);
  }
  ```
</details>

<details> 
  <summary>Prime Check (Simple)</summary>

  ### Prime Check (Simple)

  A basic algorithm to check if a number is prime.

  ```javascript
  function isPrime(n) {
      if (n <= 1) return false;
      for (let i = 2; i * i <= n; i++) {
          if (n % i === 0) return false;
      }
      return true;
  }
  ```
</details>

<details> 
  <summary>Fibonacci Sequence</summary>
  
  ### Fibonacci Sequence
  
  Generates the nth Fibonacci number iteratively.
  
  ```javascript
  function fibonacci(n) {
      if (n <= 1) return n;
      let a = 0, b = 1;
      for (let i = 2; i <= n; i++) {
          [a, b] = [b, a + b];
      }
      return b;
  }
  ```
</details>

<details> 
  <summary>Power Function (Exponentiation by Squaring)</summary>
  
  ### Power Function (Exponentiation by Squaring)

  Efficiently calculates base^exp.

  ```javascript
  function power(base, exp) {
      if (exp === 0) return 1;
      const half = power(base, Math.floor(exp / 2));
      return exp % 2 === 0 ? half * half : half * half * base;
  }
  ```
</details>

<details> 
  <summary>Binary Exponentiation (Modular)</summary>

  ### Binary Exponentiation (Modular)

  Computes (base^exp) % mod using efficient binary exponentiation.

  ```javascript
  function modPow(base, exp, mod) {
      let result = 1;
      base = base % mod;
      
      while (exp > 0) {
          if (exp % 2 === 1) result = (result * base) % mod;
          exp = Math.floor(exp / 2);
          base = (base * base) % mod;
      }
      
      return result;
  }
  ```
</details>
