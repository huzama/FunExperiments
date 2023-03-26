import traceback
import sys
  
# declaring and assigning array
A = [1, 2, 3, 4]
  
# exception handling
try:
    value = A[5]

except Exception as ex:
    # printing stack trace
    #print(ex)
    pp = traceback.format_exc()

    print(pp)
    raise ex
