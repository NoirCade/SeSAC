def insertion_sort(arr):
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # key보다 큰 원소들을 오른쪽으로 이동
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1

        # key를 적절한 위치에 삽입
        arr[j + 1] = key
        print(arr)

# 예제
arr = [12, 11, 13, 5, 6]
insertion_sort(arr)