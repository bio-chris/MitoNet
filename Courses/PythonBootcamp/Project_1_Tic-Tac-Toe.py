def board():
    board = []
    i = 1
    while i <= 9:
        board.append(i)
        i+=1
    return board

board = board()


def draw_board(board):
    i= 0
    while i <= len(board):
        if i == len(board):
            break
        print(board[i], board[i + 1], board[i + 2])
        i += 3

def set_x_1():

    global board
    #if set_position_1 > 9 and set_position_1 < 0:

    while True:
        x = input("Player1 - Enter the position where you want to place your X: ")

        try:
            x = int(x)
        except ValueError:
            print("Not an integer!")
            continue

        if int(x) <= 9 and int(x) >= 1:
            if board[int(x) - 1] == "X" or board[int(x) - 1] == "O":
                print("This position is already marked")
                continue
            board[int(x) - 1] = "X"
            break
        print("Position is outside the range")

    return board

def set_o_2():

    global board
    #if set_position_1 > 9 and set_position_1 < 0:

    while True:
        x = input("Player2 - Enter the position where you want to place your O: ")

        try:
            x = int(x)
        except ValueError:
            print("Not an integer!")
            continue

        if int(x) <= 9 and int(x) >= 1:
            if board[int(x) - 1] == "X" or board[int(x) - 1] == "O":
                print("This position is already marked")
                continue
            board[int(x) - 1] = "O"
            break
        print("Position is outside the range")

    return board


def win():
    if board[0:3] == ["X","X","X"] or board[0:3] == ["O","O","O"]:
        return True
    elif board[3:6] == ["X", "X", "X"] or board[3:6] == ["O", "O", "O"]:
        return True
    elif board[6:9] == ["X", "X", "X"] or board[6:9] == ["O", "O", "O"]:
        return True
    elif board[0:8:3] == ["X", "X", "X"] or board[0:8:3] == ["O", "O", "O"]:
        return True
    elif board[1:8:3] == ["X", "X", "X"] or board[1:8:3] == ["O", "O", "O"]:
        return True
    elif board[2:9:3] == ["X", "X", "X"] or board[2:9:3] == ["O", "O", "O"]:
        return True
    elif board[0:9:4] == ["X", "X", "X"] or board[0:9:4] == ["O", "O", "O"]:
        return True
    elif board[2:8:2] == ["X", "X", "X"] or board[2:8:2] == ["O", "O", "O"]:
        return True
    else:
        return False


def draw(board):
    count=0
    for i in board:
        if i == "X" or i == "O":
            count+=1
    if count == 9:
        return True


# Tic-Tac-Toe Game (1st Milestone Project - Python Bootcamp)

print("A Game of Tic-tac-toe")
draw_board(board)
count = 1
while True:

    set_x_1()
    draw_board(board)
    if win() == True:
        print("Player 1 has won the game")
        break

    if count == 5:
        print("Draw")
        break

    set_o_2()
    draw_board(board)
    if win() == True:
        print("Player 2 has won the game")
        break


    count+=1