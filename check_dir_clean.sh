#!/bin/bash
#
# Nagios/Icinga plugin to monitor directory for stale or excessive files.
#
# Usage:
#   check_directory.sh -d /path/to/dir -w 5:10 -c 15:20
#
# Where:
#   -w/--warning: age_minutes:file_count (e.g., "5:10" = warn if files older than 5 min OR more than 10 files)
#   -c/--critical: age_minutes:file_count (e.g., "15:20" = critical if files older than 15 min OR more than 20 files)
#
# Exit codes:
#   0 = OK
#   1 = WARNING
#   2 = CRITICAL
#   3 = UNKNOWN

# Nagios exit codes
OK=0
WARNING=1
CRITICAL=2
UNKNOWN=3

# Default stability window (seconds) - files modified within this window are ignored
STABILITY_SECONDS=2

usage() {
    cat << EOF
Usage: $0 -d DIRECTORY -w AGE:COUNT -c AGE:COUNT [-s SECONDS]

Options:
  -d, --directory DIR     Directory to monitor
  -w, --warning AGE:COUNT Warning threshold (e.g., "5:10")
  -c, --critical AGE:COUNT Critical threshold (e.g., "15:20")
  -s, --stability SECONDS  Ignore files modified within this many seconds (default: 2)
  -h, --help              Show this help message

Examples:
  $0 -d /var/spool/uploads -w 5:10 -c 15:20
    Warn if files older than 5 min OR more than 10 files
    Critical if files older than 15 min OR more than 20 files
EOF
    exit $UNKNOWN
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            DIRECTORY="$2"
            shift 2
            ;;
        -w|--warning)
            WARNING_THRESHOLD="$2"
            shift 2
            ;;
        -c|--critical)
            CRITICAL_THRESHOLD="$2"
            shift 2
            ;;
        -s|--stability)
            STABILITY_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "UNKNOWN: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DIRECTORY" || -z "$WARNING_THRESHOLD" || -z "$CRITICAL_THRESHOLD" ]]; then
    echo "UNKNOWN: Missing required parameters"
    usage
fi

# Check if directory exists
if [[ ! -d "$DIRECTORY" ]]; then
    echo "UNKNOWN: Directory does not exist: $DIRECTORY"
    exit $UNKNOWN
fi

# Parse thresholds
parse_threshold() {
    local threshold="$1"
    if [[ ! "$threshold" =~ ^([0-9]+):([0-9]+)$ ]]; then
        echo "UNKNOWN: Invalid threshold format: $threshold (expected AGE:COUNT)"
        exit $UNKNOWN
    fi
    AGE="${BASH_REMATCH[1]}"
    COUNT="${BASH_REMATCH[2]}"
}

parse_threshold "$WARNING_THRESHOLD"
WARN_AGE=$AGE
WARN_COUNT=$COUNT

parse_threshold "$CRITICAL_THRESHOLD"
CRIT_AGE=$AGE
CRIT_COUNT=$COUNT

# Validate thresholds
if [[ $CRIT_AGE -lt $WARN_AGE || $CRIT_COUNT -lt $WARN_COUNT ]]; then
    echo "UNKNOWN: Critical thresholds must be >= warning thresholds"
    exit $UNKNOWN
fi

# Get current time in seconds since epoch
NOW=$(date +%s)
STABILITY_CUTOFF=$((NOW - STABILITY_SECONDS))

# Initialize counters
FILE_COUNT=0
OLDEST_AGE=0
OLDEST_FILE=""

# Process files in directory
while IFS= read -r -d '' file; do
    # Get file modification time
    MTIME=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)
    
    if [[ -z "$MTIME" ]]; then
        continue
    fi
    
    # Skip files that are still being written (modified very recently)
    if [[ $MTIME -gt $STABILITY_CUTOFF ]]; then
        continue
    fi
    
    # Calculate age in minutes
    AGE_SECONDS=$((NOW - MTIME))
    AGE_MINUTES=$((AGE_SECONDS / 60))
    
    # Count this file
    ((FILE_COUNT++))
    
    # Track oldest file
    if [[ $AGE_MINUTES -gt $OLDEST_AGE ]]; then
        OLDEST_AGE=$AGE_MINUTES
        OLDEST_FILE=$(basename "$file")
    fi
done < <(find "$DIRECTORY" -maxdepth 1 -type f -print0 2>/dev/null)

# Check critical conditions (OR logic)
if [[ $OLDEST_AGE -gt $CRIT_AGE || $FILE_COUNT -gt $CRIT_COUNT ]]; then
    REASONS=()
    [[ $OLDEST_AGE -gt $CRIT_AGE ]] && REASONS+=("oldest file age ${OLDEST_AGE} min > ${CRIT_AGE} min")
    [[ $FILE_COUNT -gt $CRIT_COUNT ]] && REASONS+=("file count ${FILE_COUNT} > ${CRIT_COUNT}")
    
    MSG="CRITICAL: $(IFS=' AND '; echo "${REASONS[*]}")"
    [[ -n "$OLDEST_FILE" ]] && MSG="${MSG} | files=${FILE_COUNT} oldest_age=${OLDEST_AGE}min oldest_file=${OLDEST_FILE}"
    echo "$MSG"
    exit $CRITICAL
fi

# Check warning conditions (OR logic)
if [[ $OLDEST_AGE -gt $WARN_AGE || $FILE_COUNT -gt $WARN_COUNT ]]; then
    REASONS=()
    [[ $OLDEST_AGE -gt $WARN_AGE ]] && REASONS+=("oldest file age ${OLDEST_AGE} min > ${WARN_AGE} min")
    [[ $FILE_COUNT -gt $WARN_COUNT ]] && REASONS+=("file count ${FILE_COUNT} > ${WARN_COUNT}")
    
    MSG="WARNING: $(IFS=' AND '; echo "${REASONS[*]}")"
    [[ -n "$OLDEST_FILE" ]] && MSG="${MSG} | files=${FILE_COUNT} oldest_age=${OLDEST_AGE}min oldest_file=${OLDEST_FILE}"
    echo "$MSG"
    exit $WARNING
fi

# All good
MSG="OK: ${FILE_COUNT} file(s)"
[[ -n "$OLDEST_FILE" ]] && MSG="${MSG}, oldest is ${OLDEST_AGE} min"
MSG="${MSG} | files=${FILE_COUNT} oldest_age=${OLDEST_AGE}min"
echo "$MSG"
exit $OK