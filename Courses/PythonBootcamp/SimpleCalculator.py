"""
Simple Calculator 

"""

count=0
while True:


    if count == 0:

        entry1 = input("\nEnter 1st number: ")
        operation = input("\nEnter mathematical operator (+,-,*,/,^): ")
        entry2 = input("\nEnter 2nd number: ")

        try:
            float(entry1)
            float(entry2)
        except:
            print("\nNo Numbers were entered for Entry 1 and 2\n")
            continue

        if operation == "+":
            print("\n",entry1,operation,entry2, "=" ,float(entry1)+float(entry2))
            result=float(entry1)+float(entry2)
        elif operation == "-":
            print("\n",entry1,operation,entry2, "=" ,float(entry1)-float(entry2))
            result=float(entry1)-float(entry2)
        elif operation == "*":
            print("\n",entry1,operation,entry2, "=" ,float(entry1)*float(entry2))
            result=float(entry1)*float(entry2)
        elif operation == "/":
            print("\n",entry1,operation,entry2, "=" ,float(entry1)/float(entry2))
            result=float(entry1)/float(entry2)
        elif operation == "^":
            print("\n",entry1,operation,entry2, "=" ,float(entry1)**float(entry2))
            result= float(entry1)**float(entry2)
        else:
            print("No known mathematical operator was entered")
            continue

    if count > 0:

        print("\n 1st number is:", result)
        operation = input("\nEnter mathematical operator (+,-,*,/,^): ")
        entry2 = input("\nEnter 2nd number: ")

        if operation == "+":
            print("\n",result,operation,entry2, "=" ,float(result)+float(entry2))
            result=float(result)+float(entry2)
        elif operation == "-":
            print("\n",result,operation,entry2, "=" ,float(result)-float(entry2))
            result=float(result)-float(entry2)
        elif operation == "*":
            print("\n",result,operation,entry2, "=" ,float(result)*float(entry2))
            result=float(result)*float(entry2)
        elif operation == "/":
            print("\n",result,operation,entry2, "=" ,float(result)/float(entry2))
            result=float(result)/float(entry2)
        elif operation == "^":
            print("\n",result,operation,entry2, "=" ,float(result)**float(entry2))
            result= float(result)**float(entry2)
        else:
            print("No known mathematical operator was entered")
            continue


    cont = input("\nPerform new calculation (n), use result of previous calculation (p) or stop program (s)? ")


    if cont == "n":
        count=0
        continue
    elif cont == "p":
        count+=1
        continue
    elif cont == "s":
        break


