from abc import ABC, abstractmethod



class Parent(ABC):

    @abstractmethod
    def f(self):
        print('parent doing things')


class Child(Parent):

    def f(self):
        super().f()
        print('child doing something')


kid = Child()

kid.f()