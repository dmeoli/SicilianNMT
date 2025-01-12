#!/usr/bin/env perl

##  Copyright 2019 Eryk Wdowiak
##  
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##  
##      http://www.apache.org/licenses/LICENSE-2.0
##  
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

use strict;
use warnings;

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  input and output files
my $sc_infile = "dataset/train-mparamu_v2-lemmatized.sc";
my $sc_otfile = "dataset/train-mparamu_v3-lemmatized.sc.tsv";

my $en_infile = "dataset/train-mparamu_v2-lemmatized.en";
my $en_otfile = "dataset/train-mparamu_v3-lemmatized.en.tsv";

my @all_files = ([ $sc_infile, $sc_otfile ],
    [ $en_infile, $en_otfile ]);

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  for each pair of files
foreach my $file_pair (@all_files) {
    my $infile = ${$file_pair}[0];
    my $otfile = ${$file_pair}[1];

    ##  hold the words in an array
    my @otwords;

    ##  read it in
    open(INFILE, $infile) || die "could not open $infile";
    while (<INFILE>) {
        chomp;
        my $line = $_;
        my @words = split(/ /, $line);
        push(@otwords, '<BOS>', @words, '<EOS>');
    }
    close INFILE;

    ##  print it out
    open(OTFILE, ">$otfile") || die "could not overwrite $otfile";
    my $line = join("\t", @otwords);
    print OTFILE $line;
    close OTFILE;
}
