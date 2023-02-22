#!/usr/bin/env perl

##  Sicilian_Translator/extract-text/02_wrap-text.pl

##  Copyright 2020 Eryk Wdowiak
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
no warnings "uninitialized";

my @infiles = (
    "as01_en.txt", "as01_sc.txt",
    "as02_en.txt", "as02_sc.txt",
    "as03_en.txt", "as03_sc.txt",
    "as04_05_en.txt", "as04_05_sc.txt",
    "as06_en.txt", "as06_sc.txt",
    "as07_en.txt", "as07_sc.txt",
    "as08_1_en.txt", "as08_1_sc.txt",
    "as09_1_en.txt", "as09_1_sc.txt",
    "as09_2_en.txt", "as09_2_sc.txt",
    "as10_1_en.txt", "as10_1_sc.txt",
    "as10_2en.txt", "as10_2_sc.txt",
    "as11_1_en.txt", "as11_1_sc.txt",
    "as11_2_en.txt", "as11_2_sc.txt",
    "as12_en.txt", "as12_sc.txt",
    "as13_en.txt", "as13_sc.txt",
    "as14_en.txt", "as14_sc.txt",
    "as15_en.txt", "as15_sc.txt",
    "as16_en.txt", "as16_sc.txt",
    "as17_en.txt", "as17_sc.txt",
    "as18_en.txt", "as18_sc.txt",
    "as19_en.txt", "as19_sc.txt",
    "as20_en.txt", "as20_sc.txt",
    "as21_en.txt", "as21_sc.txt",
    "as22_en.txt", "as22_sc.txt",
    "as23_en.txt", "as23_sc.txt",
    "as24_en.txt", "as24_sc.txt",
    "as25_en.txt", "as25_sc.txt",
    "as26_en.txt", "as26_sc.txt",
    "as27_en.txt", "as27_sc.txt",
    "as28_en.txt", "as28_sc.txt",
    "as29_en.txt", "as29_sc.txt",
    "as30_en.txt", "as30_sc.txt",
    "as31_en.txt", "as31_sc.txt",
    "as32_en.txt", "as32_sc.txt",
    "as33_en.txt", "as33_sc.txt",
    "as34_en.txt", "as34_sc.txt",
    "as35_en.txt", "as35_sc.txt",
    "as36_en.txt", "as36_sc.txt",
    "as37_en.txt", "as37_sc.txt",
    "as38_en.txt", "as38_sc.txt",
    "as39_en.txt", "as39_sc.txt",
    "as40_en.txt", "as40_sc.txt",
);

##  clean and wrap the raw text of each file
foreach my $infile (@infiles) {

    (my $otfile = $infile) =~ s/\.txt/-wrap.txt/;
    open(INFILE, "output_txt/$infile") || die "could not open $infile";
    open(OTFILE, ">wrapped/$otfile") || die "could not overwrite $otfile";

    while (<INFILE>) {
        chomp;
        my $line = $_;

        ##  strip excess periods
        #$line =~ s/\.\.\.\.\.+/ /g;

        ##  strip excess space
        $line =~ s/\s+/ /g;

        ##  add a single space to end of line
        $line .= " ";

        ##  fix apostrophes
        $line =~ s/‘/'/g;
        $line =~ s/’/'/g;

        ##  fix double quotes
        $line =~ s/“/"/g;
        $line =~ s/”/"/g;

        ##  clean dot
        $line =~ s/•//g;

        ##  add new line after period, exclaim or question
        $line =~ s/\.(")? /\.$1\n/g;
        $line =~ s/\!(")? /\!$1\n/g;
        $line =~ s/\?(")? /\?$1\n/g;

        ##  print it out
        print OTFILE $line;
    }

    close OTFILE;
    close INFILE;
}
