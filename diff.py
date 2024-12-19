
def compare_files_method(file1_path, file2_path):
    
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False
        return next(f1, None) is None and next(f2, None) is None


def main():
    file1 = 'generation_100_ref.txt'
    file2 = '' # Put the 100th generation output path here
    
    try:

        result = compare_files_method(file1, file2)
        print(f"result: {'Files are identical' if result else 'Files are different'}")
        
        
    except FileNotFoundError:
        print("One or both files not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
