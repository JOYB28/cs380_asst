#include<iostream>
#include<list>

// ************************************************************************** //
//                          List Container Tutorial                      
// 
// CS380 Introduction to Computer Graphics
// by Eunji Hong, 2023.
//
// This file contains a tutorial for the list container in C++.
// It can be used as a reference for the keyframe list in the assignment.
//
// You can compile this file using the command:
//     g++ -std=c++11 list_tutorial.cpp -o list_tutorial
// You can run this file using the command after the compilation:
//     ./list_tutorial
//
// ************************************************************************** //

int main(){
    // Create an empty list of integers
    std::list<int> l1; // empty list of ints
    std::list<int> l2 = std::list<int>(); // empty list of ints


    // Create a list of integers with initial values
    std::list<int> l3 = {1, 2, 3, 4, 5}; // list initialized to values 1-5
    std::list<int> l4 = std::list<int>(5, 77); // list initialized to 5 values of 1
    std::list<int> l5 = std::list<int>(l3); // list initialized to a copy of l3



    // Add elements to the list
    // - list.push_back(value) : add value to the end of the list
    l1.push_back(1); // l1 = {1}
    l1.push_back(2); // l1 = {1, 2}
    l1.push_back(3); // l1 = {1, 2, 3}

    // - list.push_front(value) : add value to the front of the list
    l1.push_front(4); // l1 = {4, 1, 2, 3}

    // - list.insert(iterator, value) : add value before the iterator
    l2.insert(l2.begin(), 1); // l2 = {1}
    l2.insert(l2.begin(), 2); // l2 = {2, 1}
    l2.insert(l2.begin(), 3); // l2 = {3, 2, 1}



    // Remove elements from the list
    // - list.pop_back() : remove the last element of the list
    l1.pop_back(); // l1 = {4, 1, 2}

    // - list.pop_front() : remove the first element of the list
    l1.pop_front(); // l1 = {1, 2}

    // - list.erase(iterator) : remove the element at the iterator
    l2.erase(l2.begin()); // l2 = {2, 1}
    l2.erase(l2.begin()); // l2 = {1}



    // Access elements in the list
    // - list.front() : get the first element of the list
    int first = l3.front(); // first = 1

    // - list.back() : get the last element of the list
    int last = l3.back(); // last = 5



    // Remove all elements from the list
    // - list.clear() : remove all elements from the list
    l3.clear(); // l3 = {}



    // Check if the list is empty
    // - list.empty() : return true if the list is empty
    bool empty = l3.empty(); // empty = true



    // Get the size of the list
    // - list.size() : return the size of the list
    int size = l3.size(); // size = 0



    // Iterate through the list
    // - list.begin() : an iterator of the first element of the list
    // - list.end() : an iterator of the **next** of the last element of the list
    //              ※ Note that the value of this iterator (list.end()) is undefined and
    //              ※ the iterator of the last element is --list.end() not list.end().
    // - *it : get the value of the list entity that the iterator points to
    // - it++ : move the iterator to the next entity of the list
    // - it-- : move the iterator to the previous entity of the list
    for (std::list<int>::iterator it = l5.begin(); it != l5.end(); it++) {
        std::cout << *it << std::endl;
    }


    // Access elements in the list using an iterator
    // - list.begin() : an iterator of the first element of the list
    std::list<int>::iterator it = l5.begin(); // it = iterator of the first element of l5
    int first_it = *it; // first = 1
    std::cout << first_it << std::endl;

    it++; // it = iterator of the second element of l5
    int second_it = *it; // second = 2
    std::cout << second_it << std::endl;


    // - list.end() : an iterator of the **next** of the last element of the list
    it = l5.end(); // it = iterator of the next of the last element of l5

    it--; // it = iterator of the last element of l5
    int last_it = *it; // last = 5
    std::cout << last_it << std::endl;

    it--; // it = iterator of the second last element of l5
    int second_last_it = *it; // second_last = 4
    std::cout << second_last_it << std::endl;

}