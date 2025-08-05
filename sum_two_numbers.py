"""
sum_two_numbers.py

This script takes two numbers as input from the user and returns their sum.
It includes error handling to ensure that the inputs are valid numbers.
"""

def get_number(prompt):
    """
    Get a number from user input.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        float: The number entered by the user.

    Raises:
        ValueError: If the input is not a valid number.
    """
    while True:
        try:
            number = float(input(prompt))
            return number
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def sum_two_numbers(num1, num2):
    """
    Calculate the sum of two numbers.

    Args:
        num1 (float): The first number.
        num2 (float): The second number.

    Returns:
        float: The sum of num1 and num2.
    """
    return num1 + num2

def main():
    """
    Main function to execute the sum of two numbers.
    """
    print("Welcome to the Sum Calculator!")
    num1 = get_number("Enter the first number: ")
    num2 = get_number("Enter the second number: ")
    
    result = sum_two_numbers(num1, num2)
    print(f"The sum of {num1} and {num2} is: {result}")

if __name__ == "__main__":
    main()