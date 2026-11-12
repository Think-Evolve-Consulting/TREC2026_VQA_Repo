#! /usr/bin/perl

# Given 2 sentences from the command line, returns a score representing how similar

# this is an API service http://swoogle.umbc.edu/SimService/api.html
# PAPER : Lushan Han, Abhay L. Kashyap, Tim Finin, James Mayfield and Johnathan Weese, UMBC_EBIQUITY-CORE: Semantic Textual Similarity Systems, 
# Proc. 2nd Joint Conf. on Lexical and Computational Semantics, Association for Computational Linguistics, June 2013. 

#Usage: get.semantic.similarity.pl -sen children is playing football -sen kids are playing soccer

my $host="swoogle.umbc.edu";
my $protocol="http";


(undef, $s1, $s2) = split /-sen/," @ARGV";

$score = `wget "$protocol://$host/StsService/GetStsSim?operation=api&phrase1=$s1&phrase2=$s2" --user-agent='NIST - TRECVID' -q -O - 2>&1` ;

print "$score";









