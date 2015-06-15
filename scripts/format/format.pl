
use strict;
use warnings;

use Text::CSV_XS;
use IO::File;
use JSON::XS;
use Carp::Assert;

binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');

sub collapseTree {
    my ($tree, $msg) = @_;
    $msg = '' unless (defined($msg));

    # remove intermediate nodes that have only one child.
    my $stack = [[$tree, undef]];
    while (defined(my $tmp = shift(@$stack))) {
	my ($node, $parent) = @$tmp;

	next unless (defined($node->{'children'}));
	assert(!defined($node->{'featureList'}));

	if (scalar(keys(%{$node->{'children'}})) == 1) {
	    my $child = (values(%{$node->{'children'}}))[0];
	    if (defined($child->{'featureList'})) {
		printf STDERR ("%s isolated child: %s <- %s\n", $msg, $node->{'label'}, $child->{'label'});
		delete($parent->{'children'}->{$node->{'label'}});
		$parent->{'children'}->{$child->{'label'}} = $child;
	    } else {
		# kill intermediate node
		printf STDERR ("%s skip intermediate: %s <- %s\n", $msg, $node->{'label'}, $child->{'label'});
		$node->{'children'} = $child->{'children'};
		unshift(@$stack, [$node, $parent]);
	    }
	    next;
	}
	while ((my ($clabel, $cnode) = each(%{$node->{'children'}}))) {
	    push(@$stack, [$cnode, $node]);
	}
    }
}

sub calcFamilyCoverage {
    my ($tree, $msg) = @_;
    $msg = '' unless (defined($msg));

    while ((my ($name, $familyTree) = each(%{$tree->{'children'}}))) {
	my $featureCountList = &_calcFamilyCoverageMain($familyTree, []);
	my $covCount = scalar(grep { $_ > 0 } (@$featureCountList));
	$familyTree->{'familyCoverage'} = $covCount / scalar(@$featureCountList);
    }
}

sub _calcFamilyCoverageMain {
    my ($node, $featureCountList) = @_;

    if (defined($node->{'children'})) {
	assert(!defined($node->{'featureList'}));
	while ((my ($clabel, $cnode) = each(%{$node->{'children'}}))) {
	    &_calcFamilyCoverageMain($cnode, $featureCountList);
	}
    } else {
	for (my $i = 0, my $iL = scalar(@{$node->{'featureList'}}); $i < $iL; $i++) {
	    unless (defined($featureCountList->[$i])) {
		$featureCountList->[$i] = 0;
	    }
	    if ($node->{'featureList'}->[$i] >= 0) {
		$featureCountList->[$i]++;
	    }
	}
    }
    return $featureCountList;
}


# 
# usage: perl format.pl langlist.out language.modified.csv classifications tree.json features.json
#
my ($langlistFile, $walsFile, $classFile, $treeFile, $featureFile, $walsTreeFile) = @ARGV;
my $json = JSON::XS->new->allow_nonref->pretty;
my $tree = {
    'label' => 'ROOT',
    'children' => {},
};
my $walsTree = {
    'label' => 'ROOT',
    'children' => {},
};
my $langList = {};
my $fid2struct = [];

# ISO code corresponds to multiple WALS items
my $dupList = {};
my $badList = {};
{
    my $f = IO::File->new($langlistFile) or die;
    $f->binmode(':utf8');
    while((my $line = $f->getline)) {
	chomp($line);
	my ($iso, $count) = split(/\t/, $line);
	if ($count > 1) {
	    $dupList->{$iso} = $count;
	}
    }
}

{
    my $f = IO::File->new($classFile) or die;
    $f->binmode(':utf8');
    while((my $line = $f->getline)) {
	chomp($line);
	my ($lang, $raw) = split(/\t/, $line);
	unless ($raw) {
	    printf STDERR ("skip empty classification: %s\n", $lang);
	    next;
	}
	if ($raw eq 'Deaf sign language'
	    or $raw =~ /^Mixed language/
	    or $raw =~ /^Creole/
	    or $raw =~ /Unclassified/) {
	    printf STDERR ("skip for bad classification: %s\t%s\n", $lang, $raw);
	    $badList->{$lang} = 1;
	    next;
	}
	# next if ($raw eq 'Language isolate');

	my $leaf = { 'label' => $lang };
	$langList->{$lang} = $leaf;
	my $node = $tree;
	if ($raw ne 'Language isolate') {
	    # Language isolate -> ROOT's child
	    foreach my $internal (split(/\, /, $raw)) {
		my $child = $node->{'children'}->{$internal};
		unless (defined($child)) {
		    $child = $node->{'children'}->{$internal} = {
			'label' => $internal,
			'children' => {},
		    };
		}
		$node = $child;
	    }
	}
	$node->{'children'}->{$lang} = $leaf;
	if (defined($dupList->{$lang})) {
	    $leaf->{'children'} = {}
	}
    }
    $f->close;
}


{
    my $f = IO::File->new($walsFile) or die;
    $f->binmode(':utf8');
    my $csv = Text::CSV_XS->new( { binary => 1 } ) or die "Cannot use CSV: ".Text::CSV->error_diag();
    {
	# the first line
	my $row = $csv->getline($f);
	splice(@$row, 0, 8); # from wals_code to family
	printf STDERR ("%d features in the database\n", scalar(@$row));
	foreach my $feature (@$row) {
	    push(@$fid2struct, {
		'name' => $feature,
		'vid2label' => [],
		'label2vid' => {},
		 });
	}
    }
    my ($total, $covered) = (0, 0);
    while ((my $row = $csv->getline($f))) {
	my $wals = shift(@$row);
	my $iso = shift(@$row);
	my $glotto = shift(@$row);
	my $name = shift(@$row);
	my $latitude = shift(@$row);
	my $longtitude = shift(@$row);
	my $genus = shift(@$row);
	my $family = shift(@$row);

	my ($label, $leaf);
	if ($dupList->{$iso}) {
	    $label = $iso . '_' . $wals;

	    printf STDERR ("ISO ambiguity: use %s\n", $label);

	    my $tmpLeaf = $langList->{$iso};
	    $leaf = { 'label' => $label };
	    $tmpLeaf->{'children'}->{$label} = $leaf;
	} else {
	    $label = $iso;
	    $leaf = $langList->{$iso};
	}

	# calc coverage
	for (my $i = 0, my $iL = scalar(@$row); $i < $iL; $i++) {
	    $total++;
	    $covered++ if ($row->[$i]);
	}

	unless (defined($leaf)) {
	    unless (defined($badList->{$iso})) {
		printf STDERR ("skip %s (undefined)\n", $iso);
	    }
	    next;
	}
	$leaf->{'wals'} = $wals;
	$leaf->{'iso'} = $iso;
	$leaf->{'name'} = $name;
	$leaf->{'glotto'} = $glotto;
	$leaf->{'latitude'} = $latitude;
	$leaf->{'longtitude'} = $longtitude;
	$leaf->{'featureList'} = [];
	for (my $i = 0, my $iL = scalar(@$row); $i < $iL; $i++) {
	    my $feature = $row->[$i];
	    if ($feature) {
		my $fval = $fid2struct->[$i]->{'label2vid'}->{$feature};
		unless (defined($fval)) {
		    $fval = (split(/ /, $feature))[0] - 1;
		    $fid2struct->[$i]->{'label2vid'}->{$feature} = $fval;
		    $fid2struct->[$i]->{'vid2label'}->[$fval] = $feature;
		}
		push(@{$leaf->{'featureList'}}, $fval);
	    } else {
		push(@{$leaf->{'featureList'}}, -1);
	    }
	}

	if (defined($walsTreeFile)) {
	    my $familyNode = $walsTree->{'children'}->{$family};
	    unless(defined($familyNode)) {
		$familyNode = $walsTree->{'children'}->{$family} = {
		    'label' => $family,
		    'children' => {},		    
		};
	    }
	    my $genusNode = $familyNode->{'children'}->{$genus};
	    unless (defined($genusNode)) {
		$genusNode = $familyNode->{'children'}->{$genus} = {
		    'label' => $genus,
		    'children' => {},		    
		};
	    }
	    $genusNode->{'children'}->{$label} = {
		'label' => $label,
		'iso' => $iso,
		'wals' => $wals,
		'name' => $name,
		'glotto' => $glotto,
		'latitude' => $latitude,
		'longtitude' => $longtitude,
		'featureList' => \@{$leaf->{'featureList'}},
	    };
	    
	}
    }
    $f->close;
    printf STDERR ("feature coverage: %d / %d (%f)\n", $covered, $total, $covered / $total);
}


&collapseTree($tree, 'Ethnologue');
if (defined($walsTreeFile)) {
    &collapseTree($walsTree, 'WALS');
}

&calcFamilyCoverage($tree, 'Ethnologue');
if (defined($walsTreeFile)) {
    &calcFamilyCoverage($walsTree, 'WALS');
}


{
    my $f;
    if (defined($treeFile) and $treeFile ne '-') {
	$f = IO::File->new($treeFile, 'w') or die;
	$f->binmode(':utf8');
    } else {
	$f = \*STDOUT;
    }
    $f->printf("%s\n", $json->encode($tree));
    $f->close if ($f->isa('IO::File'));
}
{
    my $f;
    if (defined($featureFile) and $featureFile ne '-') {
	$f = IO::File->new($featureFile, 'w') or die;
	$f->binmode(':utf8');
    } else {
	$f = \*STDOUT;
    }
    $f->printf("%s\n", $json->encode($fid2struct));
    $f->close if ($f->isa('IO::File'));
}

if (defined($walsTreeFile)) {
    my $f;
    if (defined($walsTreeFile) and $walsTreeFile ne '-') {
	$f = IO::File->new($walsTreeFile, 'w') or die;
	$f->binmode(':utf8');
    } else {
	$f = \*STDOUT;
    }
    $f->printf("%s\n", $json->encode($walsTree));
    $f->close if ($f->isa('IO::File'));
}


# use Dumpvalue;
# Dumpvalue->new->dumpValue($tree);
# Dumpvalue->new->dumpValue($fid2struct);
# exit;
