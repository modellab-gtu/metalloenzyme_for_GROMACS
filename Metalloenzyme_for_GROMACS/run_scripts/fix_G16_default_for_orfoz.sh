
target=$(grep -rl "%Mem=3000MB" "$AMBERHOME/lib/python3."*/site-packages/pymsmt/mol/gauio.py 2>/dev/null)

if [ -n "$target" ]; then
    echo "Found gauio.py at: $target"
    cp "$target" "${target}.bak"
    sed -i 's/Mem=3000MB/Mem=220GB/g' "$target"
    echo "Memory updated to 220GB. Backup saved as ${target}.bak"
else
    echo "Error: gauio.py with '%Mem=3000MB' not found under $AMBERHOME/lib/python3.*"
fi


