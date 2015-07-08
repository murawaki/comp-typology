
use strict;
use warnings;

use Text::CSV;
use IO::File;

my $langList = &load(\*STDIN);

foreach my $lang (sort { $a cmp $b } (keys(%$langList))) {
    printf("%s\t%d\n", $lang, $langList->{$lang});
}


sub load {
    my ($istream) = @_;

    my $csv = Text::CSV->new( { binary => 1 } ) or die "Cannot use CSV: ".Text::CSV->error_diag();
    my $langList = {};
    {
	# the first line
	my $row = $csv->getline($istream);
    }

    while ((my $row = $csv->getline($istream))) {
	my $lang = $row->[1];
	if ($lang) {
	    $langList->{$lang}++;
	} else {
	    printf STDERR ("no ISO code:\t%s\n", $row->[0]);
	}
    }
    return $langList;
}

