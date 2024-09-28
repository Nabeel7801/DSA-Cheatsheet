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
  <summary>KMP Pattern Matching (Prefix Function)</summary>

  ### KMP Pattern Matching (Prefix Function)

  Efficient string searching algorithm (Knuth-Morris-Pratt).

  ```javascript
  Copy code
  function kmpPrefixFunction(s) {
      const prefix = Array(s.length).fill(0);
      for (let i = 1, j = 0; i < s.length; i++) {
          while (j > 0 && s[i] !== s[j]) j = prefix[j - 1];
          if (s[i] === s[j]) j++;
          prefix[i] = j;
      }
      return prefix;
  }
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
  ```
</details>

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
  Copy code
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
  Copy code
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
  Copy code
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

<details> 
  <summary>nCr (Combinations)</summary>

  ### nCr (Combinations)

  Calculates the number of combinations (n choose r).

  ```javascript
  Copy code
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

