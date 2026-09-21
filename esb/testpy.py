def main():
    print("Hello, World!")
    i_declare_in_main = "This variable is declared in the main function."
    inner_function()  # Call the inner function to demonstrate scope.

def inner_function():
    print("This is an inner function.")
    i_declare_in_inner = "This variable is declared in the inner function."
    print(i_declare_in_inner)
    print(name)  # This will raise an error because name is not in scope.

if __name__ == "__main__":
    name = "i am a variable in the main scope in __name__ == '__main__' block"
    main()