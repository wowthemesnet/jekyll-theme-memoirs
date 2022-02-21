---
layout: post
title:  "Reflection on OOP"
author: yusuf
categories: oop
image: assets/images/featured_image/7575.png
published: True
date: 2022-02-19
---

In a forum, students were sharing stories about system failures that cause big consequences. One story tells about how the failed system is causing [the loss of $450 mil. in just 45 minutes][1]. It is general knowledge to buy a stock when the price is low, and sell when the price is high. Unfortunately, what's being deployed to the production is doing exactly the opposite of it. In addition, an investigation found that this is also happened due to no code review or QA process before the deployment.

Another interesting story that was shared is coming from a well-known company in the aviation industry, Boeing. A faulty in the new model's system was causing two fatal crashes in just five months. Similar to the previous incident, [lack of documentation and the improper software development process - rushed release][2], contributed as the cause of this incident.

[According to SAP][3], being object-oriented means that the focus of the development is on objects that represent abstracts or concrete things in the real world, defined by their characteristics and properties. Each object is encapsulated, making its information hidden from another object and reusable.

One thing that should always be remembered from this course is that object can be anything. It's not only about classes in programming language, even a database can be an object in object-oriented. Just like a class having one or more relationship with other classes, a table in a database also possible to have a relationship with another table. Although, the definitions and the type of relationships are _different_ in Class diagram and ERD.

In object-oriented system design, one of the advantages is to make it easier to troubleshoot problems. It means that being object-oriented helps us in making sure that the system will behave according to the expectations.

In order to do that, the trick is to test its logic in different ways: **documentation**, **source code**, and **unit test**.

To summarize, it's like building a human-like figure out of Lego. First, we create the left part of the leg. Then we make sure that it is working properly. And then we go with building the right part of the leg, we can reuse the design of the left leg. But since we want the right part to have a different pose, we then create the improvement and test it again. The leg and the body part is exposed only to things they are required or supposed to, like joints between stomach and thigh, which makes the joints of the foot is hidden from the body.

From this course, it gave me several new knowledge that can be used in my day to day activity at work. I have never done a logic test in my documentation before, and I've been doing an improper _unit test_ all this time. It turns out, all those three are strongly related to each other, despite they are totally different.

So, the to-do list for me to keep practicing:
1. More practice on making diagrams
2. More practice on unit testing when developing anything

Acquiring this knowledge can help me to progress in my career, as it should improve the quality of my work. I think it also exposes me to managerial path of engineering, a position which usually does the design rather than executing the development of the design. Combined with my existing technical skill, they are probably more attractive in the labor market.

[1]: https://www.bugsnag.com/blog/bug-day-460m-loss
[2]: https://c2y6x2t8.rocketcdn.me/wp-content/uploads/2019/09/the-boeing-737-max-saga-lessons-for-software-organizations.pdf
[3]: https://help.sap.com/doc/saphelp_nw73/7.3.16/en-US/c3/225b5654f411d194a60000e8353423/content.htm?no_cache=true