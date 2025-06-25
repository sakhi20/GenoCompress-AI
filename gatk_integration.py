#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GATK integration for genomic data compression pipeline.
Provides functionality for variant calling and quality control.
"""

import os
import subprocess
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GATKConfig:
    gatk_path: str = 'gatk'
    reference_genome: Optional[str] = None
    tmp_dir: str = 'tmp'
    java_options: str = '-Xmx4g -Xms2g'
    keep_intermediate: bool = False

class GATKProcessor:
    def __init__(self, config: GATKConfig):
        self.config = config
        self._validate_gatk_installation()
        
    def _validate_gatk_installation(self):
        """Check if GATK is properly installed and accessible."""
        try:
            result = subprocess.run(
                [self.config.gatk_path, '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"GATK not found or not working: {result.stderr}")
            logger.info(f"GATK version: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError("GATK executable not found. Please install GATK or specify correct path.")

    def _run_gatk_command(self, command_list):
        """
        Helper function to execute GATK commands using java -jar.
        """
        gatk_jar_path = os.path.join(os.path.dirname(os.path.dirname(self.config.gatk_path)), 'share', 'gatk4-4.6.1.0-0', 'gatk-package-4.6.1.0-local.jar')
        java_command = ['java'] + self.config.java_options.split() + ['-jar', gatk_jar_path] + command_list

        logger.info(f"Running command: {' '.join(java_command)}")
        try:
            subprocess.run(java_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"GATK command failed: {e.stderr}")
            raise

    def index_fasta(self, fasta_path: str) -> None:
        """Create index files for a FASTA file."""
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
            
        # Create dictionary
        dict_path = fasta_path.replace('.fasta', '.dict')
        self._run_gatk_command([
            'CreateSequenceDictionary',
            '-R', fasta_path,
            '-O', dict_path
        ])
        
        # Create index
        self._run_gatk_command([
            'IndexFeatureFile',
            '-I', fasta_path
        ])

    def align_to_reference(self, fastq_path, output_bam_path):
        """
        Align sequences in FASTQ format to the reference genome using BWA and add read groups.
        """
        sample_name = os.path.splitext(os.path.basename(fastq_path))[0]

        bwa_command = [
            'bwa', 'mem',
            self.config.reference_genome,
            fastq_path
        ]
        samtools_command = [
            'samtools', 'view', '-S', '-b', '-',
            '-o', output_bam_path + '.unsorted.bam'
        ]

        logger.info(f"Aligning {fastq_path} with BWA...")
        bwa_process = subprocess.Popen(bwa_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        samtools_process = subprocess.Popen(samtools_command, stdin=bwa_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        bwa_process.stdout.close()
        stdout, stderr = samtools_process.communicate()

        if samtools_process.returncode != 0:
            logger.error(f"Error during alignment: {stderr}")
            raise Exception(f"Alignment failed for {fastq_path}")
        else:
            logger.info(f"Alignment successful: {output_bam_path}.unsorted.bam")

        # Add read group information
        add_rg_command = [
            'AddOrReplaceReadGroups',
            '-I', output_bam_path + '.unsorted.bam',
            '-O', output_bam_path + '.rg.bam',
            '-RGLB', 'lib1',
            '-RGPL', 'ILLUMINA',
            '-RGPU', 'unit1',
            '-RGSM', sample_name,  # Changed from '-RGSN' to '-RGSM'
            '-RGID', 'rg1'
        ]
        self._run_gatk_command(add_rg_command)
        os.rename(output_bam_path + '.rg.bam', output_bam_path + '.sorted.bam')
        os.remove(output_bam_path + '.unsorted.bam') # Remove the unsorted BAM
        logger.info(f"Read groups added to: {output_bam_path}.sorted.bam")

        # Sort the BAM file
        sort_command = [
            'SortSam',
            '-I', output_bam_path + '.sorted.bam',
            '-O', output_bam_path, # Output to the final BAM path
            '-SORT_ORDER', 'coordinate'
        ]
        self._run_gatk_command(sort_command)
        os.remove(output_bam_path + '.sorted.bam') # Remove the intermediate sorted BAM
        logger.info(f"BAM file sorted: {output_bam_path}")

        # Index the BAM file
        index_command = [
            'BuildBamIndex',
            '-I', output_bam_path,
            '-O', output_bam_path + '.bai'
        ]
        self._run_gatk_command(index_command)
        logger.info(f"BAM index created: {output_bam_path}.bai")

    def call_variants(self, input_bam, output_vcf):
        """
        Call variants using GATK HaplotypeCaller.
        """
        command = [
            'HaplotypeCaller',  # Just the GATK tool name
            '-R', self.config.reference_genome,
            '-I', input_bam,
            '-O', output_vcf,
            '--native-pair-hmm-threads', '4'  # Adjust threads as needed
        ]
        try:
            self._run_gatk_command(command)
            logger.info(f"Variants called successfully: {output_vcf}")
        except Exception as e:
            logger.error(f"Calling variants: GATK command failed: {e}")
            raise

    def compare_variants(self, original_vcf: str, reconstructed_vcf: str, output_report: str) -> str:
        """
        Compare variants between original and reconstructed sequences.
        
        Args:
            original_vcf: VCF from original sequences
            reconstructed_vcf: VCF from reconstructed sequences
            output_report: Path for output comparison report
            
        Returns:
            Path to comparison report
        """
        self._run_gatk_command([
            'Concordance',
            '-R', self.config.reference_genome,
            '--truth', original_vcf,
            '--evaluation', reconstructed_vcf,
            '--summary', output_report
        ])
        
        return output_report

    def calculate_concordance_metrics(self, original_bam: str, reconstructed_bam: str) -> dict:
        """
        Calculate concordance metrics between original and reconstructed BAM files.
        
        Args:
            original_bam: Path to original BAM
            reconstructed_bam: Path to reconstructed BAM
            
        Returns:
            Dictionary with concordance metrics
        """
        # Create intermediate files
        original_vcf = os.path.join(self.config.tmp_dir, 'original.vcf')
        reconstructed_vcf = os.path.join(self.config.tmp_dir, 'reconstructed.vcf')
        report_file = os.path.join(self.config.tmp_dir, 'concordance_report.txt')
        
        # Call variants for both files
        self.call_variants(original_bam, original_vcf)
        self.call_variants(reconstructed_bam, reconstructed_vcf)
        
        # Compare variants
        self.compare_variants(original_vcf, reconstructed_vcf, report_file)
        
        # Parse report
        metrics = self._parse_concordance_report(report_file)
        
        # Clean up if needed
        if not self.config.keep_intermediate:
            os.remove(original_vcf)
            os.remove(original_vcf + '.idx')
            os.remove(reconstructed_vcf)
            os.remove(reconstructed_vcf + '.idx')
            os.remove(report_file)
            
        return metrics

    def _parse_concordance_report(self, report_path: str) -> dict:
        """Parse GATK concordance report into a dictionary."""
        metrics = {}
        try:
            with open(report_path, 'r') as f:
                for line in f:
                    if line.startswith('PPV'):
                        metrics['PPV'] = float(line.split()[-1])
                    elif line.startswith('Sensitivity'):
                        metrics['Sensitivity'] = float(line.split()[-1])
                    elif line.startswith('Specificity'):
                        metrics['Specificity'] = float(line.split()[-1])
                    elif line.startswith('F1_Score'):
                        metrics['F1_Score'] = float(line.split()[-1])
        except Exception as e:
            logger.error(f"Error parsing concordance report: {e}")
            
        return metrics
