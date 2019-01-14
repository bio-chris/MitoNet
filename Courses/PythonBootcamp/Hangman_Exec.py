"""

Classic Hangman Game

"""


from random_words import RandomWords
rw = RandomWords()


while True:

    word = rw.random_words()
    word_list = list(word[0])

    print("The word you are looking for has", len(word[0]), "letters \n")
    print("You have", len(word[0])*3 ,"guesses to find the word you are looking for\n")

    print(" _ " * len(word[0]))
    blank = ["_"]*len(word[0])


    count=1

    while count <= len(word[0])*3:

        print("\nGuess", count, "\n",)
        letter = input("Enter any letter from a to z: ").lower()


        if not letter.isalpha():
            print("\nNo letter was entered")
            continue

        if letter in word_list:

            index = word_list.index(letter)
            blank[index] = letter
            word_list[index] = "_"

            print(" ".join(blank))


            guess_word = input("\nEnter full word if you think you know it. Press Enter to otherwise continue : ")

            if guess_word == word[0]:

                print("\nYou have found the word. You Win!")
                break

            if guess_word == "":
                pass

            elif guess_word != word[0]:

                print("\nThis is not the word we are looking for. Sorry!")



        elif letter not in word_list:

            print(letter, "is not in the word")

        if "_" not in blank:
            print("\nYou guessed all letters correct. You Win!")
            break

        count += 1


    if count == len(word[0])*3+1:
        print("\nYou have exceeded", len(word[0]) * 3, "rounds without guessing the word. You have lost.\n")
        print("\nThe word you were looking for was", word[0])
        new_round = input("\nEnter c for another round or press s to stop the game: ")
        if new_round == "c":
            continue
        if new_round == "s":
            break

    new_round = input("\nEnter c for another round or press s to stop the game: ")
    if new_round == "c":
        continue
    if new_round == "s":
        break



