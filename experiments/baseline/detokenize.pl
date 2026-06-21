#!/usr/bin/env perl
# Detokenize English MT output back to natural text (rejoin punctuation, capitalize),
# using the Napizia module, so it can be scored against the raw reference.
#   PERL5LIB=perl-module perl experiments/baseline/detokenize.pl < hyp.tok.en > hyp.en
use strict;
use warnings;
use Napizia::Translator;

while (my $line = <STDIN>) {
    chomp $line;
    $line = en_detokenizer($line);
    $line = en_capitalize($line);
    print "$line\n";
}
