# iMessage Database Converter

A comprehensive Python tool for converting iMessage SQLite databases to phone-style HTML reports and PowerPoint presentations with advanced analysis and forensic capabilities.

**Supports both database formats:**
- `chat.db` - from Mac computers or iTunes/Finder backups
- `sms.db` - from iPhone direct extraction or forensic tools

## 🚀 Features

### Core Functionality
- **iMessage Database Conversion**: Convert `chat.db` or `sms.db` SQLite files to visually appealing HTML reports
- **PowerPoint Export**: Generate presentation-ready slides with optimal message framing
- **Contact Resolution**: Integrate with both Mac AddressBook (AddressBook-v22.abcddb) and iPhone AddressBook (AddressBook.sqlitedb) for displaying contact names instead of phone numbers
- **Phone-Style UI**: Recreate the familiar iPhone message interface in HTML
- **Progress Indicators**: Real-time progress bars for long-running operations (processing messages, generating frames, hash calculations)
- **HTML Pagination**: Split large conversations into multiple pages with navigation and overview index
  - Automatic folder creation with index.html and individual page files
  - Configurable messages per page (e.g., 500, 1000)
  - Navigation between pages with Previous/Next links
  - Overview page with links to all pages showing message ranges and dates
  - Respects frame settings for each page
- **Attachment Embedding**: Automatically embed images and videos directly in HTML reports
  - Images (PNG, JPEG, GIF, HEIC) embedded as base64
  - HEIC images automatically converted to JPEG for browser compatibility
  - Videos under 10MB embedded with HTML5 player
  - MOV files shown with clickable links to open in media player
  - Magic byte detection for files without extensions
  - Removes attachment placeholder characters from message text

### Advanced Filtering & Search
- **Text Search**: Find messages containing specific text (case-insensitive)
- **Date Range Filtering**: Filter messages by specific date/time ranges
- **Chat-Specific Views**: Focus on individual conversations

### Analytics & Forensics
- **Communication Heatmaps**: Visualize messaging patterns by time of day and day of week
- **Participant Analysis**: Response time analysis and activity metrics
- **Interaction Matrix**: Who talks to whom frequency analysis
- **Relationship Graph**: Interactive HTML graph visualizing participant relationships using Cytoscape.js
  - Node-based visualization showing all participants
  - Edge weights representing message frequency between participants
  - Configurable filtering (minimum messages, top participants)
  - Optional exclusion of "Me" from the graph
  - Top 10 relationships statistics
- **Deleted Message Detection**: Identify gaps and potential deleted messages
- **Hash Verification**: Cryptographic integrity verification for forensic use
- **Epoch Timestamp Info**: Detailed timestamp conversion audit trails

## 🔧 Installation

### Required Dependencies

```bash
pip install tqdm
```

### Optional Dependencies (for enhanced features)

```bash
# For PowerPoint export
pip install python-pptx pillow

# For HEIC image support
pip install pillow-heif

# For older Python versions (< 3.9)
pip install backports.zoneinfo
```

## 💻 Usage

### Graphical User Interface (GUI)

Launch the GUI for easy point-and-click operation:

```bash
python messages_db_contacts.py \
  --db chat.db \
  --powerpoint presentation.pptx \
  --show-participant-info \
  --show-interaction-matrix \
  --show-communication-heatmap
```

**Forensic analysis with full verification:**
```bash
python messages_db_contacts.py \
  --db chat.db \
  --out forensic_report.html \
  --show-hash-verification \
  --show-epoch-info \
  --show-deleted-info \
  --show-attachments
```

**Embed actual images and videos from attachments folder:**
```bash
python messages_db_contacts.py \
  --db chat.db \
  --out messages_with_attachments.html \
  --show-attachments \
  --attachments-folder ~/Library/Messages/Attachments
```

**Create paginated HTML output for large conversations:**
```bash
python messages_db_contacts.py \
  --db sms.db \
  --addressbook AddressBook.sqlitedb \
  --out conversation.html \
  --messages-per-page 500 \
  --per-frame 50
```
This creates a folder named `conversation` containing:
- `index.html` - Overview page with all analysis and links to all pages
- `page_1.html`, `page_2.html`, etc. - Individual pages with 500 messages each, split into frames of 50 messages

**Generate an interactive relationship graph:**
```bash
python messages_db_contacts.py \
  --db chat.db \
  --addressbook AddressBook-v22.abcddb \
  --out messages.html \
  --relations-graph-out relationships.html \
  --relations-min-edge 10 \
  --relations-top-nodes 100
```
This creates an interactive HTML graph visualizing participant relationships with configurable filtering.

## 📊 Parameter Reference

### Required Parameters
| Parameter | Description |
|-----------|-------------|
| `--db` | Path to the iMessage SQLite database (chat.db or sms.db) |
| `--out` | Path for output HTML file |

### Optional Parameters

#### Input/Output Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--addressbook` | None | Path to AddressBook database for contact resolution |
| `--attachments-folder` | None | Path to folder containing attachments (embeds images/videos) |
| `--powerpoint`, `--pptx` | None | Generate PowerPoint instead of HTML |

#### Filtering Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chat-id` | None | Filter to specific chat ID |
| `--contact-name` | None | Friendly name for the chat |
| `--date-from` | None | Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) |
| `--date-to` | None | End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) |
| `--search` | None | Search for messages containing specific text |

#### Display Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--width` | 375 | Phone frame width in pixels |
| `--font-size` | 14 | Font size for message text |
| `--title` | "IMESSAGE MESSAGES" | Report title |
| `--per-frame` | 0 | Messages per phone frame (0 = all in one) |
| `--messages-per-page` | 0 | Messages per HTML page (0 = no pagination). Creates folder with index.html and page files |

#### Timezone Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tz-from` | UTC | Source timezone for timestamps |
| `--tz-to` | None | Destination timezone (e.g., America/Chicago) |
| `--show-tz-in-messages` | False | Show timezone in each timestamp |
| `--show-phone-numbers` | False | Show phone numbers with contact names |

#### Analysis Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--show-epoch-info` | False | Show epoch timestamp information |
| `--show-deleted-info` | False | Show deleted message analysis |
| `--show-participant-info` | False | Show participant activity analysis |
| `--show-interaction-matrix` | False | Show who talks to whom matrix |
| `--show-communication-heatmap` | False | Show time/day activity heatmaps |
| `--show-hash-verification` | False | Show hash verification report |
| `--show-attachments` | False | Show attachment info (file paths, types, stickers) |

#### Relationship Graph Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--relations-graph-out` | None | Path to write an HTML graph of participant relationships (Cytoscape.js) |
| `--relations-min-edge` | 5 | Only include edges with at least this many messages |
| `--relations-top-nodes` | 150 | Keep at most this many strongest participants |
| `--relations-exclude-me` | False | Exclude 'Me' from the relationship graph |

#### Interface Options
| Parameter | Description |
|-----------|-------------|
| `--gui` | Launch graphical user interface |

## 📈 Output Features

### HTML Report Structure
- **Phone-style interface** with message bubbles
- **Participant information** with response time analysis
- **Interactive timeline** with date/time stamps
- **Paginated output** (optional) - For large conversations:
  - Clean overview page (index.html) with all analysis and statistics
  - Individual message pages with navigation (Previous/Next)
  - Each page shows date range and message count
  - All pages respect frame settings for optimal viewing
- **Embedded media** - Images and videos displayed inline
  - PNG, JPEG, GIF, HEIC images embedded directly
  - Small videos (MP4) play in browser
  - MOV files with clickable links
  - Automatic file type detection (no extension needed)
- **Attachment metadata** showing file paths, mime types, and sticker indicators
- **Search highlighting** when using search functionality
- **Timezone conversion** indicators
- **Hash verification** tables for forensic validation

### PowerPoint Presentation
- **Optimized slide layout** with automatic message framing
- **Professional formatting** suitable for presentations
- **Embedded analytics** charts and graphs
- **Frame numbering** for easy navigation

### Analytics Reports
- **Communication patterns** by hour/day
- **Response time analysis** between participants
- **Message frequency trends** over time
- **Participant interaction matrices**
- **Relationship graph** - Interactive network visualization showing:
  - Participant nodes sized by activity level
  - Edges weighted by message frequency
  - Interactive pan/zoom controls
  - Top relationships statistics
  - Configurable filtering options
- **Deleted message insights**

## 🔒 Privacy & Security

### Data Handling
- **Local processing only** - no data sent to external servers
- **Read-only database access** - original files are never modified
- **Hash verification** ensures data integrity
- **Timezone audit trails** for forensic applications

### Forensic Considerations
- **Cryptographic hashes** (SHA-256) for integrity verification
- **Epoch timestamp preservation** with conversion audit trails
- **Deleted message analysis** without data recovery
- **Chain of custody** information in reports

## 🐛 Troubleshooting

### Common Issues

**"Database is locked" error:**
- Ensure Messages app is closed
- Check file permissions
- Copy database to a different location

**Which database file to use:**
- **Mac/iTunes backup**: Use `chat.db` (located in ~/Library/Messages/ on Mac)
- **iPhone extraction**: Use `sms.db` (from iPhone filesystem or forensic tools)
- Both formats are supported - the tool works with either database

**"No module named 'zoneinfo'" error:**
- Update to Python 3.9+ or install backport: `pip install backports.zoneinfo`

**Contact names not appearing:**
- Verify AddressBook database path
- Check file permissions
- Ensure AddressBook database is from the same user account

**PowerPoint export fails:**
- Install required dependencies: `pip install python-pptx pillow`
- Check output directory permissions

**HEIC images not displaying:**
- Install HEIC support: `pip install pillow-heif`
- HEIC images will be automatically converted to JPEG for browser compatibility

**Attachments not showing:**
- Ensure `--show-attachments` flag is set
- Provide `--attachments-folder` path to embed actual media files
- Check that attachment files exist at the specified location

### Performance Tips
- **Large databases**: Use date filtering to reduce processing time
  - Progress bars show real-time status for message processing, frame generation, and hash calculations
- **Large conversations**: Use `--messages-per-page` to create paginated output for faster browser loading
  - Recommended: 500-1000 messages per page for optimal performance
  - Helps browsers load and render large conversations smoothly
  - Progress bars track page generation for paginated output
- **Memory usage**: Process smaller date ranges for very large conversations
- **GUI responsiveness**: Use CLI for batch processing of multiple files
- **Attachments**: Processing attachments can take time - progress bars help track the operation

## 📝 Examples & Use Cases

### Personal Use
- **Archive conversations** before device upgrades
- **Create memory books** from important conversations
- **Search message history** for specific information
- **Analyze communication patterns** with family/friends

### Professional Use
- **Legal discovery** with forensic hash verification
- **Digital forensics** with timeline analysis
- **Communication auditing** for compliance
- **Research and analysis** of communication patterns

### Technical Use
- **Database analysis** and exploration
- **Timestamp conversion** between timezones
- **Data integrity verification** for forensic purposes
- **Custom reporting** and analytics

## 🤝 Contributing

This tool is designed to be extensible. Areas for potential enhancement:
- Additional export formats (PDF, CSV, JSON)
- Enhanced analytics and visualizations
- Support for other messaging platforms
- Advanced search capabilities (regex, boolean operators)
- Integration with other forensic tools

## ⚖️ Legal Notice

This tool is intended for legitimate use cases including:
- Personal data archival and analysis
- Legal discovery with proper authorization
- Digital forensics by qualified professionals
- Research with appropriate permissions

**Important**: Ensure you have proper authorization before accessing and analyzing messaging databases that are not your own.

## 📄 License

This project is provided as-is for educational and professional use. Please respect privacy and legal requirements when using this tool.