#lang racket



;;; The code is divided into
;;; Part 1 - Neuron Network Functions
;;; Part 2 - Layer Functions
;;; Part 3 - Mlp Functions
;;; Part 4 - Calculation Functions
;;; Part 5 - Testing with XOR Function
;;; Part 6 - Testing with Simple-MNIST Dataset



;;; Part 1 - Neuron Network Functions

;; Create a new neuron
(define (new-neuron)
  (let ((theta (rand-theta))
        (backward '())
        (cache #f)
        (trained #f)
        (train-sum 0))
    (lambda ([method 'activate] [arg '()])
      (cond
       ((eq? method 'backward)
        (push! (list arg (rand-weight)) backward))
       ((eq? method 'set)
        (set! cache arg))
       ((eq? method 'reset)
        (set! cache #f)
        (set! trained #f)
        (set! train-sum 0))
       ((eq? method 'sum)
        (set! train-sum (+ train-sum arg)))
       ((eq? method 'list)
        (map (lambda (el) (cadr el)) backward))
       ((eq? method 'train)
        (if (not trained)
            (set! backward (train backward (* cache (- 1 cache) train-sum)))
            (set! trained #t)))
       ((eq? method 'activate)
        (if cache
            cache
            (begin
              (set! cache (sigmoid (- (sum-weight backward) theta)))
              cache)))))))

;; Train neurons
(define (train lst err)
  (if (empty? lst)
      '()
      (let ((n (caar lst))
            (w (cadar lst)))
        (n 'sum (* err w))
        (cons (list n (+ w (* (n) err)))
              (train (cdr lst) err)))))



;;; Part 2 - Layer Functions

;; Create a new neuron layer
(define (new-layer n)
  (if (= n 0) '()
    (cons (new-neuron) (new-layer (- n 1)))))

;; Link two layers together DDD
(define (link-layers left right)
  (if (or (empty? left) (empty? right))
    '()
    (begin
      (map (lambda (e) ((car right) 'backward e)) left)
      (link-layers left (cdr right)))))

;; Set a layer of neurons (activated/inactivated)
(define (set-layer layer in)
  (if (empty? layer) '()
      (begin
        ((car layer) 'set (car in))
        (set-layer (cdr layer) (cdr in)))))

;; Reset a single layer (inactivate it)
(define (reset-layer layer)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'reset)
        (reset-layer (cdr layer)))))

;; Run 'train on each neuron in layer
(define (train-layer layer)
  (if (empty? layer)
    '()
    (begin
      ((car layer) 'train)
      (train-layer (cdr layer)))))

;; Used in 'run-mlp & 'train-mlp
(define (run-layer layer)
  (if (empty? layer) '()
      (cons ((car layer)) (run-layer (cdr layer)))))

;; Used in 'train-mlp procedure
(define (sum-layer layer out desired a)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'sum (* a (- (car desired) (car out))))
        (cons (car out)
              (sum-layer (cdr layer)
                           (cdr out)
                           (cdr desired)
                           a)))))

;; Train each layer in reversed mlp
(define (train-layers rev-mlp)
  (if (empty? rev-mlp)
      '()
      (begin
        (train-layer (car rev-mlp))
        (train-layers (cdr rev-mlp)))))



;;; Part 3 - Mlp Functions

;; Create new mlp
(define (new-mlp spec)
  (let ((mlp (map new-layer spec)))
    (link-mlp mlp)
    mlp))

;; Link up layers in an unlinked mlp
(define (link-mlp mlp)
  (if (= (length mlp) 1) '()
      (begin
        (link-layers (car mlp) (cadr mlp))
        (link-mlp (cdr mlp)))))

;; Reset each neuron in mlp (inactivate it)
(define (reset-mlp mlp)
  (if (empty? mlp)
      '()
      (begin
        (reset-layer (car mlp))
        (reset-mlp (cdr mlp)))))

;; Receive the output of mlp
(define (run-mlp mlp in)
  (set-layer (car mlp) in)
  (let ((out (run-layer (last mlp))))
    (reset-mlp mlp)
    out))

;; Train mlp
(define (train-mlp mlp in desired [a 1])
  (set-layer (car mlp) in)
  (let ((out (run-layer (last mlp))))
    (sum-layer (last mlp) out desired a)
    (train-layers (reverse mlp))
    (reset-mlp mlp)
    out))



;;; Part 4 - Calculation Functions

;; Sigmoid function
(define (sigmoid x)
  (/ (+ 1.0 (exp (- x)))))

;; Return a new random weight in (-1.2, 1.2)
(define (rand-weight) 
  (- (* (random) 1.2) 0.6))

;; Return a new random threshold in (-1.2, 1.2)
(define (rand-theta)
  (- (* (random) 1.2) 0.6))

;; Define push! syntax
(define-syntax push!
  (syntax-rules ()
    ((push item place)
     (set! place (cons item place)))))

;; Sum weight
(define (sum-weight 1st)
  (if (empty? 1st) 0
      (+ (* ((caar 1st)) (cadar 1st))
         (sum-weight (cdr 1st)))))

;; Round to binary 1's and 0's
(define (round-output out)
  (map (compose inexact->exact round) out))



;;; Part 5 - Testing with XOR Function

;; Construct mlp
(define mlp (new-mlp '(2 8 1)))

;; Train mlp. Print #t if it succeeds.
(do ((i 1 (+ i 1)))
    ((> i 10000) #t)
  (train-mlp mlp '(0 0) '(0))
  (train-mlp mlp '(0 1) '(1))
  (train-mlp mlp '(1 0) '(1))
  (train-mlp mlp '(1 1) '(0)))

;; Test mlp
(run-mlp mlp '(0 0))
(run-mlp mlp '(0 1))
(run-mlp mlp '(1 0))
(run-mlp mlp '(1 1))

;; Round the testing mlp result, and it should print the following
;'(0)
;'(1)
;'(1)
;'(0)
(round-output (run-mlp mlp '(0 0)))
(round-output (run-mlp mlp '(0 1)))
(round-output (run-mlp mlp '(1 0)))
(round-output (run-mlp mlp '(1 1)))





;;; Part 6 - Testing with Simple-MNIST Dataset

;; Toolbox Functions

; Transfer a decimal digit number to a 10-element binary list (e.g. 4 -> '(0 0 0 1 0 0 0 0 0 0))
(define (digit-transfer n)
  (define (digit-iter r i)
    (define flag
      (if (= i n) 1 0))
    (if (> i 9)
    	'()
    	(cons flag (digit-iter r (+ i 1)))))
  (digit-iter '() 0))

; Transfer a list of ones and zeros to a digit by returning the sequence of the first 1 (e.g. '(0 0 0 1 0 0 0 0 0 0) -> 4)
(define (digit-reverse lst)
  (define (digit-iter a-lst i)
    (if (eq? (cdr a-lst) '())
        'n
        (if (= (car a-lst) 1)
  	        i
  	        (digit-iter (cdr a-lst) (+ i 1)))))
  (digit-iter lst 0))


;; Train mlp with data in train_mnist.csv file

(require "csv-reader.rkt")

(define file-path "train_mnist.csv")
(define file-path-test "test_mnist.csv")

(define input-file (open-input-file file-path #:mode 'text ))
(define input-file-test (open-input-file file-path-test #:mode 'text ))

; Make a csv-reader
(define csv-reader (make-csv-reader #:doublequote #f
                                    #:quoting #f))

; Return a list of array lists (consisting of string data) from given input-file
(define dataset (for/list ([m (in-producer (csv-reader input-file))]
                     #:break (eof-object? m))
            m))
(define dataset-test (for/list ([m (in-producer (csv-reader input-file-test))]
                     #:break (eof-object? m))
            m))

; Create a new mlp
(define my-mlp (new-mlp '(39 40 40 10)))

; Train the mlp with dataset
(define (train-data dataset)
  ; Map dataset with train-list
  (define (train-list lst)
    ; Train the mlp one time with one list like '(5 3.13 4.21 ... 1.21)
    (train-mlp my-mlp (cdr lst) (digit-transfer (car lst))))
  (map train-list dataset))

; Test the mlp with given data
(define (test-data dataset)
  ; Map dataset with test-list
  (define (test-list lst)
    ; E.g., 3.18E+02 -> 318
    (set! lst (map (lambda(e) (string->number e 10 'read 'decimal-as-inexact)) lst))
    ; Test the mlp
    (round-output (run-mlp my-mlp (cdr lst))))
  (map test-list dataset))

;; Implement training & testing

; Set dataset to an numerical version, e.g.,
; from '("5" "3.13" "4.21" ... "1.21") 
; to   '( 5   3.13   4.21  ...  1.21 )
(set! dataset
  (map (lambda(lst)
         ; E.g., 3.18E+02 -> 318
         (set! lst (map (lambda(e) (string->number e 10 'read 'decimal-as-inexact)) lst))
         lst) dataset))

; Train mlp for many times
(do ((i 1 (+ i 1)))
        ((> i 100) #t)
  (train-data dataset) (write i))


; Test mlp with dataset
; If you want to see the binary version (a 10 * 10 matrix), uncomment the following line
;(test-data  dataset-test)

; If you want to see the decimal version (recommended), uncomment the following line
(map digit-reverse (test-data  dataset-test))



