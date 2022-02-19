---
layout: post
title:  "Designing Class Diagram for Self Service Checkout System with Object-Oriented Design"
author: yusuf
categories: oop
image: assets/images/featured_image/self-service-diagram-supermarket.png
published: True
date: 2022-01-17
---

In one assignment, I have to design a Class diagram with a scenario of a self-service checkout system for a supermarket. The following is the scenario that must be fulfilled by the diagram that I will create.

```text
1. When an item is scanned, it is added to the customers purchases in a virtual shopping basket. 
   A. Items can either be scanned with the barcode reader or they can be weighed using the integrated scales. 
   B. When the customer has finished scanning their items, they will be given the option to scan their loyalty card, if they have one. 
2. The customer then purchases their items using several payment methods (e.g., cash, card, Apple Pay, AndroidPay, vouchers, loyalty points).
3. Supermarket staff will have the ability to override transactions where required
   A. They will also be responsible for checking the age of a customer for restricted items.
4. A completed transaction will update the supermarkets stock control systems and generate an alert for warehouse staff if either the shelves should be replenished or if stock needs to be reordered.
```

# I. The Diagram

Based on the scenario, here is the diagram that I made, followed by a table explaining why I chose these classes for the diagram as well as their relationships.

![](../assets/images/post_image/supermarket-class-diagram.png)

## I.A. Explanations

| Class Name | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Loyalty    | <br/>**S**ince every customer will enrol in a loyalty program, where each customer will have one loyalty card. And considering that the loyalty can be accessed without the presence of the customer. A class is created that will represent the Loyalty object.                                                                                                                                                                                                                                                              |
| Customer   | <br/>**A**s the main Actor in the scenario, the Customer class will represent the buyer’s actor. For every Customer will have one Loyalty object, linked by its `loyalty_id` attribute. This object will be able to `scan_items()`, `pay_order()`, `enter_loyalty_card()`, and see their loyalty information through the `get_loyalty_number()`.                                                                                                                                                                              |
| Prodcut    | <br/>**P**roduct class represents the product the store is selling. But to make it easy, no matter how many items there are in the inventory, the system will be having only one object for all of them. For example, there are 10 books and 5 bottles in the inventory. The system will have only 2 objects of the Product class that represents the books and bottles. Instead, the information regarding the number of stocks available will be stored as the class attribute.                                             |
| Order      | <br/>**A**lthough an Order is not a living thing, an order can have its behaviours. Thus, constructing a class designated for orders will be helpful to make it centralize all the data and operations about it. Having its class also makes it easy for the overall system during a transaction. It will help the Customer object do a payment, it will update Product object attributes after a transaction is done, etc.                                                                                                   |
| Employee   | <br/>**T**he employee is another living actor in the scenario. Its job is to order products from suppliers, override transactions manually in the case of scanning error, or check the Customer’s age when a restricted item is in the bucket. Thus, creating an Employee class will help in placing, or encapsulating the information and operations correctly.                                                                                                                                                              |
| Payment    | <br/>**T**he Payment class will be created when a Customer object is about to pay. The reason why it is having its class is that there will be multiple payment methods available for the Customer object, and they are all having different attributes and behaviours on each method. Having a parent class for each method will help in simplifying the construction of the class. <br/><br/>There are classes that inherit Payment class that represents various method of payments available:<br/>Cash, Coupon, and Card. |

## I.B. Relathionships

### Loyalty
_**Association**_
- Customer - A customer class will have a Loyalty object with a 1-to-1 relationship. A Customer won't be able to create many Loyalty objects.

### Customer
_**Association**_
- Order - A Customer can create an order by scan_items, thus the relationship is Association as Order will closely related with Customer objects. Customers can create many Order objects, thus this association is in a 1-to-many relationship.
- Product - Since the Customer will be able to create an Order object by scanning the product, an association with the Product should exist. This relationship will help the Customer on the `scan_items()` operation. Because the `product_id` will be exposed to the Customer.

### Product
_**Aggregation**_
- Order - Since the order is made up of multiple Product objects, the relationship between the Product and Order class is an Aggregation relationship.

### Order
_**Association**_
- Payment - An order will be related to payment activity. But not all Order objects will create a Payment object. This relationship means that not all information of Payment will be stored in an Order. Through `payment_id`, it can tell which Order object is related to which Payment object.

### Employee
_**Association**_
- Product - Product class will alert Employee object if the number of stock available is low, as well as the ability for Employee to order the product – with `reorder_stock_from_supplier()` operation inside Employee object and `create_stock_alert()` operation inside Product object.
- Order - Employee object will be able to do customer's age checking as well as manually override an order, through `check_customer_age()` and `override_order()` operation.

### Payment
_**Generalization**_
- Cash, Coupon, Card - An inheritance from the Payment class that represents the sub-type of a Payment object.

### Card
_**Generalization**_
- Debit, Credit - Represents the subtype of the Card class.
