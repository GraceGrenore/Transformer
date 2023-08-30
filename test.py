students = list()

name  = input("Enter student name:")
while name != "0":
    if name in students:
        print("This student is already exsited")
    else:
        students.append(name)
    name  = input("Enter student name:")

print(students)
