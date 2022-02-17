package ru.lyovkin.lab3_bugs

class Level(bugsCount: Int) {
    lateinit var bugs : ArrayList<Bug>

    init {
        bugs = ArrayList(bugsCount)
    }
}