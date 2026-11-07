(define (domain tyreworld)
  (:types obj - object
	  tool wheel nut - obj
	  container hub - object)
(:predicates (isopen ?x - container)
             (closed ?x - container)
             (have ?x - obj)
             (in ?x - container ?y - obj)
             (loose ?x - nut ?y - hub)
             (tight ?x - nut ?y - hub)
             (unlocked ?x - container)
             (on-ground ?x - hub)
             (not-on-ground ?x - hub)
             (inflated ?x - wheel)
             (not-inflated ?x - wheel)
             (fastened ?x - hub)
             (unfastened ?x - hub)
             (free ?x - hub)
             (on ?x - wheel ?y - hub)
             (intact ?x - wheel))
(:constants wrench pump jack - tool)

(:action open
:parameters (?x - container)
:precondition (and (unlocked ?x)
                   (closed ?x))
:effect (and (isopen ?x)
   (not (closed ?x))))

(:action close
:parameters (?x - container)
:precondition (isopen ?x)
:effect (and (closed ?x)
   (not (isopen ?x))))

(:action fetch
:parameters (?x - obj  ?y - container)
:precondition (and (in ?x ?y) (isopen ?y))
:effect (and (have ?x)
   (not (in ?x ?y))))

(:action put-away
:parameters (?x - obj ?y - container)
:precondition (and (have ?x) (isopen ?y))
:effect (and (in ?x ?y)
   (not (have ?x))))

(:action loosen
:parameters (?x - nut ?y - hub)
:precondition (and (have wrench) (tight ?x ?y) (on-ground ?y))
:effect (and (loose ?x ?y)
   (not (tight ?x ?y))))

(:action tighten
:parameters (?x - nut ?y - hub)
:precondition (and (have wrench) (loose ?x ?y) (on-ground ?y))
:effect (and (tight ?x ?y)
   (not (loose ?x ?y))))

(:action jack-up
:parameters (?y - hub)
:precondition (and (on-ground ?y) (have jack))
:effect (and (not-on-ground ?y)
   (not (on-ground ?y))  (not (have jack))))

(:action jack-down
:parameters (?y - hub)
:precondition (not-on-ground ?y)
:effect (and (on-ground ?y)  (have jack)
   (not (not-on-ground ?y))))

(:action undo
:parameters (?x - nut ?y - hub)
:precondition (and (not-on-ground ?y) (fastened ?y) (have wrench) (loose ?x ?y))
:effect (and (have ?x) (unfastened ?y)
   (not (fastened ?y)) (not (loose ?x ?y))))

(:action do-up
:parameters (?x - nut ?y - hub)
:precondition (and (have wrench) (unfastened ?y) (not-on-ground ?y) (have ?x))
:effect (and (loose ?x ?y) (fastened ?y)
   (not (unfastened ?y)) (not (have ?x))))

(:action remove-wheel
:parameters (?x - wheel ?y - hub)
:precondition (and (not-on-ground ?y) (on ?x ?y) (unfastened ?y))
:effect (and (have ?x) (free ?y)
   (not (on ?x ?y))))

(:action put-on-wheel
:parameters (?x - wheel ?y - hub)
:precondition (and (have ?x) (free ?y) (unfastened ?y) (not-on-ground ?y))
:effect (and (on ?x ?y)
   (not (free ?y)) (not (have ?x))))

(:action inflate
:parameters (?x - wheel)
:precondition (and (have pump) (not-inflated ?x) (intact ?x))
:effect (and (inflated ?x)
   (not (not-inflated ?x)))))
