"""

Code below runs Blackjack game without any classes

"""

import random

"""

Deck for Blackjack

13 cards of each clubs, diamonds, hearts and spades (52 in total)
13 cards are:
Ace (11), 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack (10), Queen (10), King (10)

"""

def deck():

    global list_deck
    list_deck = []

    i = 2
    while i <= 11:
        if i == 10:
            list_deck+= [i]*16
        else:
            list_deck += [i] * 4
        i += 1
    return list_deck
###

#print(list_deck)

def Player():

    run = 1

    global blackjack
    blackjack = 0

    global money

    global bet

    global list_deck

    global stand
    stand = False

    while True:

        if run == 1:
            bet = input("Place your bet: ")

        try:
            bet = int(bet)
        except ValueError:
            print("You did not enter an integer")
            continue

        if run >= 3:

            choice = input("Enter hit to continue or stand to end your turn: ")

            if choice == "hit":
                pass

            elif choice == "stand":
                print("Your current score is: ", blackjack)
                stand = True
                break

            else:
                print("You did not enter hit nor stand")
                continue

        drawn_card = random.choice(deck())
        print(drawn_card)

        """
        Checking if card probabilities are correct

        probability = list_deck.count(drawn_card) / len(list_deck)
        print(probability)
        """

        blackjack += drawn_card
        print("Your current Score: ", blackjack)

        if blackjack > 21:
            print("Bust! Game over")
            money-=bet


            break
        if blackjack == 21:
            print("You Win!")
            money+=bet


            break

        deck().remove(drawn_card)
        #print(random.choice(list_deck))

        if run >= 52:
            break
        run += 1

    return list_deck
    return blackjack
    return money
    return bet

def Dealer():
    if stand == True:

        global money

        global list_deck

        print("Dealers turn")

        blackjack_dealer = 0

        while True:
            drawn_card = random.choice(deck())
            print(drawn_card)

            blackjack_dealer += drawn_card
            print("Dealers Score: ", blackjack_dealer)

            deck().remove(drawn_card)

            if blackjack_dealer > 21:
                print("Dealer busts! You Win!")


                money+=bet
                break
            if blackjack == 21:
                print("Dealer wins")


                money-=bet
                break


            if blackjack_dealer > 17:
                break

        if blackjack_dealer > blackjack and blackjack_dealer <= 21:
            print("You loose! ", blackjack_dealer, " > ", blackjack)


            money-=bet
        elif blackjack_dealer < blackjack and blackjack_dealer <= 21:
            print("You win! ", blackjack_dealer, " < ", blackjack)


            money+=bet
        elif blackjack_dealer == blackjack:
            print("Draw ", blackjack_dealer, " = ", blackjack)

        return money
        return list_deck



"""

Code below calls function above to play Blackjack

"""

money = 0

while True:
    Player()
    Dealer()

    print("Money on account: ", money)

    new_round = input("Enter y to play another round. Enter n to stop: ")

    if new_round == "y":
        continue
    if new_round == "n":

        if money < 0:
            print("You have lost ", -1*money)
        elif money > 0:
            print("You have won ", money)

        break
    while new_round != "y" and new_round != "n":
        new_round = input("Enter y to play another round. Enter n to stop")


    if len(list_deck) <= 6:
        break



"""

Blackjack code using classes (Taken from Milestone 2 project solution)

"""


