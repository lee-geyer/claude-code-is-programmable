# This is a test file for Claude Code to modify
# Task: Fix the function below to correctly calculate the factorial of a number

def factorial(n):
    # This function has a bug - it doesn't handle the base case correctly
    # Please fix it to properly calculate factorial
    if n < 0:
        return "Error: Factorial not defined for negative numbers"
    # Add base case for n=0 or n=1
    if n == 0 or n == 1:
        return 1
    result = n * factorial(n - 1)
    return result

# Test the function
print(factorial(5))  # Should print 120