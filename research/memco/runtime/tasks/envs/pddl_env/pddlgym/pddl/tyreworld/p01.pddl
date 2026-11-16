(define (problem tyreworld-1)
(:domain tyreworld)
(:objects
wrench jack pump - tool
the-hub1 - hub
nuts1 - nut
boot - container
r1 w1 - wheel
)
(:init
(in jack boot)
(in pump boot)
(in wrench boot)
(unlocked boot)
(closed boot)
(intact r1)
(in r1 boot)
(not-inflated r1)
(on w1 the-hub1)
(on-ground the-hub1)
(tight nuts1 the-hub1)
(fastened the-hub1)
)
(:goal
(and
(on r1 the-hub1)
(inflated r1)
(in w1 boot)
)
)
)
