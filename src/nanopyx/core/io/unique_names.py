# a list of ludicrous dog names to be used as unique identifiers
# lets see how much time it takes until someone finds this

import random

names = [
    "Poodlef",
    "Schnoodlepuff",
    "Muttley",
    "Furball",
    "Poodlefroot",
    "Hotdog",
    "Barkalicious",
    "Pupperoni",
    "Doggydoo",
    "Puppycinno",
    "Poodlepops",
    "Barkers",
    "Muttleydoo",
    "Poodlepup",
    "Pupcake",
    "Barkleberry",
    "Fluffernutter",
    "Puppykins",
    "Poodleberry",
    "Pupperpuff",
    "Poodlepop",
    "Barktastic",
    "Fluffy_McFluffernutter",
    "Sir_Barksalot",
    "Lady_Woofington",
    "Barkley_von_Wagner",
    "Count_Fluffernutter",
    "Duke_of_Woofington",
    "Baroness_Snugglesnout",
    "Sir_Wigglesworth",
    "Lady_Pawsington",
    "Biscuit_McFluffernutter",
    "Princess_Fluffybutt",
]


def intToRoman(num):
    """
    Convert an integer to a roman numeral.

    Parameters:
        num (int): The integer to be converted.

    Returns:
        str: The roman numeral representation of the input integer.
    """
    m = ["", "M", "MM", "MMM"]
    c = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM "]
    x = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    i = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]

    # Converting to roman
    thousands = m[num // 1000]
    hundreds = c[(num % 1000) // 100]
    tens = x[(num % 100) // 10]
    ones = i[num % 10]

    ans = thousands + hundreds + tens + ones

    return ans


_already_used = []


def get_unique_name() -> str:
    """
    Get a unique name from the list of names.

    >>> name = get_unique_name()
    """
    # get a unique name
    global _already_used
    name = random.choice(names) + "_" + str(intToRoman(random.randint(1, 10)))
    while name in _already_used:
        name = random.choice(names) + "_" + str(intToRoman(random.randint(1, 10)))
    _already_used.append(name)
    return name
