def FIB_No():
    Dest = 'D:/FIB_No.txt'
    FIBWrite = open(Dest,'w')
    print("Enter a no to which you want to calculate FIB: ")

    first = 0
    second = 1

    for x in range(0 , int(input())+1):
       if x is 0:
            FIBWrite.write(str(first))
            FIBWrite.write('\n')
       elif x == 1:
            FIBWrite.write(str(second))
            FIBWrite.write('\n')
       else:
            temp = first + second
            first = second
            second = temp
            FIBWrite.write(str(temp))
            FIBWrite.write('\n')

    FIBWrite.write('Bitlength is: ')
    FIBWrite.write(str(temp.bit_length()))
    FIBWrite.close()
    print('Your file has been created in destination ', Dest )

def Complex_No():
    for a in range(4):
        for b in range(4):
            i = complex(a,b)
            print(i ** 7)


complex_no = complex(6.2,66.5)
list_array = ["Diff","Data Types", 5 , 6]
tupple_list = ("Read","Only",5 ,6)
set =  {"No repetition",'No order so no index',6}
length_of_list_tupple_int = len(list_array)

def Fact(x):
    rtrn = 1

    while x != 0:
        rtrn = x * rtrn
        x += -1
    
    return rtrn

def euler():
    rtrn = 1
    for x in range(1,2000):
        rtrn += 1/Fact(x)
    return rtrn

def oeuler():
    for x in range(1,1000):
        print((1+1/x)**x)
  