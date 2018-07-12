#!/bin/bash

trim(){
    [[ "$1" =~ ^[[:space:]]*(.*[^[:space:]])[[:space:]]*$ ]]
    printf "%s" "${BASH_REMATCH[1]}"
}

# help
display_help() {
    echo "Usage: $0 [option...] " >&2
    echo
    echo "   -h, --help                 Display this help message"
    echo "   -i, --input-file           Input file name"
    echo "   -p, --ipa                  Get IPA transcription of words"
    echo "   -k, --kirshenbaum          Get Kirshenbaum transcription of words"
    echo
    # echo some stuff here for the -a or --add-options 
    exit 1
}

# As long as there is at least one more argument, keep looping
while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        # This is an arg value type option. Will catch -i value or --input-file value
        -i|--input-file)
        shift # past the key and to the value
        INPUTFILE=$1
        ;;
        # This is an arg value type option. Will catch -p value or --ipa value
        -p|--ipa)
        IPA=true
        ;;
        # This is an arg value type option. Will catch -k value or --kirshenbaum value
        -k|--kirshenbaum)
        KIRSHEN=true
        ;;
        # display help
        -h | --help)
        display_help  # Call your function
        exit 0
        ;;
        *)
        # Do whatever you want with extra options
        #echo "Unknown option '$key'"
        ;;
    esac
    # Shift after checking all the cases to get the next option
    shift
done


while read word; do
  wp="$word"
  if [ "$IPA" = true ]; then
    sIPA="$(espeak -q --ipa $word)"
    sIPA=$(trim "$sIPA")
    wp="$wp\t$sIPA"
  fi
  if [ "$KIRSHEN" = true ]; then
    sKIRSHEN="$(espeak -q -x $word)"
    sKIRSHEN=$(trim "$sKIRSHEN")
    wp="$wp\t$sKIRSHEN"
  fi
  echo -e "$wp"
done < "${INPUTFILE:-/dev/stdin}" 


