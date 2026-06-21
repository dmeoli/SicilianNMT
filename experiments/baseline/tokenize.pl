#!/usr/bin/env perl
# Faithful Sicilian/English tokenizer using the original Napizia module (the paper's
# preprocessing): lowercases, ASCII-folds accents, expands Sicilian contractions and
# conjunctive pronouns to literary forms (ô -> a lu, dû -> di lu, mû -> mi lu ...).
# This cuts Sicilian sparsity, which is the point of the paper's recipe.
#
#   PERL5LIB=perl-module perl experiments/baseline/tokenize.pl sc < in > out
#   PERL5LIB=perl-module perl experiments/baseline/tokenize.pl en < in > out
use strict;
use warnings;
use Napizia::Translator;

my $lang = $ARGV[0] // 'sc';

while (my $line = <STDIN>) {
    chomp $line;
    $line = rm_malice($line);
    $line =~ s/~~~/ /g;
    $line = ($lang eq 'sc') ? sc_tokenizer($line) : en_tokenizer($line);

    ##  rm_morejunk (from dataset/02_tokenizer.pl)
    $line =~ s/[\x{2014}\x{2013}\x{2015}]/-/g;   # em/en/horizontal dashes
    $line =~ s/[\x{0161}\x{0160}]/s/g;            # š Š
    $line =~ s/\x{2026}/ . . . /g;                # …
    $line =~ s/\x{0153}/oe/g;                     # œ
    $line =~ s/\x{00e6}/ae/g;                     # æ

    $line =~ s/\s+/ /g;
    $line =~ s/^ //;
    $line =~ s/ $//;
    print "$line\n";
}
