echo %time% start
blastp -query ..\\data\\samples_p_0.5.fasta -out ..\\data\\samples_p_0.5.xml -db proGenDB -outfmt 5 -num_threads 8 -max_target_seqs 10
echo %time% finish 0.5
blastp -query ..\\data\\samples_p_0.75.fasta -out ..\\data\\samples_p_0.75.xml -db proGenDB -outfmt 5 -num_threads 8 -max_target_seqs 10
echo %time% finish 0.75
pause