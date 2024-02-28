def prn(val):
    if isinstance(val, str):
        print('문자열 자료임')
        print('-'*100)
        print(val)
    
    elif isinstance(val, list):
        print('리스트 자료임')
        for x in val:
            print('-'*100)
            print(x)
    

if __name__ == '__main__':
    prn('3')