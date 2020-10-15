def castFloat(string):
    Float=0
    try:
        if float(string) > 0:
            Float=float(string) 
            
    except ValueError:
        pass
    return Float 

def castInt(string):
    Float=0
    try:
        if int(string) > 0:
            Float=int(string)            
    except ValueError:
        pass
    return Float 
