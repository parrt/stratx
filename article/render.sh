#!/bin/bash

I="/Users/parrt/github/stratx/article/"
O="/tmp/partial-dependence"

#while true
#do
	if test $I/css/article.css -nt $O/L2-loss.html || \
           test $I/stratpd.xml -nt $O/stratpd.html 
	then
		java -jar /Users/parrt/github/bookish/target/bookish-1.0-SNAPSHOT.jar -target html -o $O $I/article.xml
	fi
	sleep .2s
#done
