(define (problem tyreworld-10)
(:domain tyreworld)
(:objects
wrench jack pump - tool
the-hub1 the-hub2 the-hub3 - hub
nuts1 nuts2 nuts3 - nut
boot - container
r1 w1 r2 w2 r3 w3 - wheel
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
(intact r2)
(in r2 boot)
(not-inflated r2)
(intact r3)
(in r3 boot)
(not-inflated r3)
(on w1 the-hub1)
(on-ground the-hub1)
(tight nuts1 the-hub1)
(fastened the-hub1)
(on w2 the-hub2)
(on-ground the-hub2)
(tight nuts2 the-hub2)
(fastened the-hub2)
(on w3 the-hub3)
(on-ground the-hub3)
(tight nuts3 the-hub3)
(fastened the-hub3)
)
(:goal
(and
(inflated r1)
(inflated r2)
(on r3 the-hub3)
(inflated r3)
(in w3 boot)
)
)
)
